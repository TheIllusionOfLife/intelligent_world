"""LibCST-based semantic mutation operators for higher-level code transformations."""

import random

import libcst as cst


def _collect_with_visitor(tree: cst.Module, visitor: cst.CSTVisitor) -> None:
    """Walk a CST tree with a visitor using the MetadataWrapper approach."""
    try:
        wrapper = cst.metadata.MetadataWrapper(tree)
        wrapper.visit(visitor)
    except Exception:  # noqa: BLE001
        # Fall back to simple visit for cases metadata wrapper can't handle
        tree.visit(visitor)


class _ContinueFinder(cst.CSTVisitor):
    """Detect continue statements in a loop body (not nested loops)."""

    def __init__(self) -> None:
        self.found = False

    def visit_Continue(self, node: cst.Continue) -> None:
        self.found = True

    def visit_For(self, node: cst.For) -> bool:
        return False  # Don't recurse into nested loops

    def visit_While(self, node: cst.While) -> bool:
        return False  # Don't recurse into nested loops


def _body_contains_continue(node: cst.For) -> bool:
    finder = _ContinueFinder()
    try:
        node.body.visit(finder)
    except Exception:  # noqa: BLE001
        return True  # Conservatively skip if we can't analyze
    return finder.found


def mutate_guard_insertion(source: str, rng: random.Random) -> str:
    """Insert an early-return guard clause at the start of a function body."""
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return source

    class _FuncFinder(cst.CSTVisitor):
        def __init__(self) -> None:
            self.funcs: list[cst.FunctionDef] = []

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            self.funcs.append(node)

    finder = _FuncFinder()
    _collect_with_visitor(tree, finder)

    if not finder.funcs:
        return source

    target_func = rng.choice(finder.funcs)

    # Pick a parameter for the guard condition
    param_names: list[str] = []
    for param in target_func.params.params:
        if param.name and isinstance(param.name, cst.Name):
            param_names.append(param.name.value)

    if not param_names:
        return source

    guard_param = rng.choice(param_names)
    conditions = [
        f"{guard_param} is None",
        f"not {guard_param}",
        f"{guard_param} < 0",
        f"{guard_param} == 0",
    ]
    condition_str = rng.choice(conditions)
    fallback_values = ["None", "0", "()", "[]", "False", "-1"]
    fallback = rng.choice(fallback_values)

    # Build the guard as part of a dummy function, then extract the statement
    dummy_code = f"def _():\n    if {condition_str}:\n        return {fallback}\n"
    try:
        dummy_tree = cst.parse_module(dummy_code)
        dummy_func = dummy_tree.body[0]
        if not isinstance(dummy_func, cst.FunctionDef):
            return source
        dummy_body = dummy_func.body
        if not isinstance(dummy_body, cst.IndentedBlock) or not dummy_body.body:
            return source
        guard_stmt = dummy_body.body[0]
    except (cst.ParserSyntaxError, IndexError):
        return source

    class _GuardInserter(cst.CSTTransformer):
        def __init__(self, target_name: str, guard: cst.BaseCompoundStatement) -> None:
            self._target_name = target_name
            self._guard = guard
            self._done = False

        def leave_FunctionDef(
            self,
            original_node: cst.FunctionDef,
            updated_node: cst.FunctionDef,
        ) -> cst.FunctionDef:
            if self._done:
                return updated_node
            if updated_node.name.value != self._target_name:
                return updated_node
            self._done = True
            body = updated_node.body
            if isinstance(body, cst.IndentedBlock):
                new_body = body.with_changes(body=(self._guard, *body.body))
                return updated_node.with_changes(body=new_body)
            return updated_node

    try:
        new_tree = tree.visit(_GuardInserter(target_func.name.value, guard_stmt))
        return new_tree.code
    except Exception:  # noqa: BLE001
        return source


def mutate_loop_conversion(source: str, rng: random.Random) -> str:
    """Convert a for-range loop to an equivalent while loop."""
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return source

    class _ForFinder(cst.CSTVisitor):
        def __init__(self) -> None:
            self.fors: list[cst.For] = []

        def visit_For(self, node: cst.For) -> None:
            if isinstance(node.iter, cst.Call):
                func = node.iter.func
                if isinstance(func, cst.Name) and func.value == "range":
                    if len(node.iter.args) == 1 and isinstance(node.target, cst.Name):
                        if not _body_contains_continue(node):
                            self.fors.append(node)

    finder = _ForFinder()
    _collect_with_visitor(tree, finder)

    if not finder.fors:
        return source

    target_for = rng.choice(finder.fors)
    loop_var = target_for.target.value  # type: ignore[union-attr]

    # Get the limit expression as source code
    limit_arg = target_for.iter.args[0].value  # type: ignore[union-attr]
    try:
        limit_src = cst.Module(
            body=[cst.SimpleStatementLine(body=[cst.Expr(value=limit_arg)])]
        ).code.strip()
    except Exception:  # noqa: BLE001
        return source

    class _ForReplacer(cst.CSTTransformer):
        def __init__(self, target: cst.For) -> None:
            self._target_body = target.body
            self._target_iter = target.iter
            self._done = False

        def leave_For(
            self,
            original_node: cst.For,
            updated_node: cst.For,
        ) -> cst.BaseStatement | cst.FlattenSentinel[cst.BaseStatement]:
            if self._done:
                return updated_node
            # Match by comparing the body and iter objects
            if original_node.body is not self._target_body:
                return updated_node
            if original_node.iter is not self._target_iter:
                return updated_node
            self._done = True

            # Build init statement and increment
            try:
                init_stmt = cst.parse_statement(f"{loop_var} = 0\n")
                increment_stmt = cst.parse_statement(f"    {loop_var} += 1\n")
                while_test = cst.parse_expression(f"{loop_var} < {limit_src}")
            except cst.ParserSyntaxError:
                return updated_node

            # Build while body: original for body + increment
            original_body = original_node.body
            if isinstance(original_body, cst.IndentedBlock):
                new_body = original_body.with_changes(
                    body=(*original_body.body, increment_stmt),
                )
            else:
                return updated_node

            while_stmt = cst.While(
                test=while_test,
                body=new_body,
                leading_lines=original_node.leading_lines,
            )

            return cst.FlattenSentinel([init_stmt, while_stmt])

    try:
        new_tree = tree.visit(_ForReplacer(target_for))
        result = new_tree.code
        cst.parse_module(result)
        return result
    except Exception:  # noqa: BLE001
        return source


def mutate_variable_extraction(source: str, rng: random.Random) -> str:
    """Factor repeated subexpressions into a named variable."""
    try:
        tree = cst.parse_module(source)
    except cst.ParserSyntaxError:
        return source

    class _ExprCollector(cst.CSTVisitor):
        def __init__(self) -> None:
            self.expressions: list[str] = []

        def visit_BinaryOperation(self, node: cst.BinaryOperation) -> None:
            try:
                code = cst.Module(
                    body=[cst.SimpleStatementLine(body=[cst.Expr(value=node)])]
                ).code.strip()
                if len(code) > 3:
                    self.expressions.append(code)
            except Exception:  # noqa: BLE001
                pass

    collector = _ExprCollector()
    _collect_with_visitor(tree, collector)

    # Find duplicated expressions
    expr_counts: dict[str, int] = {}
    for code in collector.expressions:
        expr_counts[code] = expr_counts.get(code, 0) + 1

    duplicates = [code for code, count in expr_counts.items() if count >= 2]
    if not duplicates:
        return source

    target_expr_code = rng.choice(duplicates)
    var_name = f"_extracted_{rng.randint(0, 999)}"

    class _Extractor(cst.CSTTransformer):
        def __init__(self, expr_code: str, var: str) -> None:
            self._expr_code = expr_code
            self._var = var
            self._replacement_count = 0
            self._inserted = False

        def leave_BinaryOperation(
            self,
            original_node: cst.BinaryOperation,
            updated_node: cst.BinaryOperation,
        ) -> cst.BaseExpression:
            try:
                code = cst.Module(
                    body=[cst.SimpleStatementLine(body=[cst.Expr(value=original_node)])]
                ).code.strip()
            except Exception:  # noqa: BLE001
                return updated_node
            if code == self._expr_code:
                self._replacement_count += 1
                return cst.Name(value=self._var)
            return updated_node

        def leave_IndentedBlock(
            self,
            original_node: cst.IndentedBlock,
            updated_node: cst.IndentedBlock,
        ) -> cst.IndentedBlock:
            if self._replacement_count == 0 or self._inserted:
                return updated_node
            self._inserted = True
            try:
                assign_stmt = cst.parse_statement(f"    {self._var} = {self._expr_code}\n")
            except cst.ParserSyntaxError:
                return updated_node
            return updated_node.with_changes(body=(assign_stmt, *updated_node.body))

    try:
        new_tree = tree.visit(_Extractor(target_expr_code, var_name))
        result = new_tree.code
        cst.parse_module(result)
        return result
    except Exception:  # noqa: BLE001
        return source
