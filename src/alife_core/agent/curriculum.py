from alife_core.models import EvaluationResult, RunConfig


def should_unlock_next_task(result: EvaluationResult, config: RunConfig) -> bool:
    return (
        result.train_pass_ratio >= config.pass_ratio_threshold
        and result.fitness >= config.fitness_threshold
    )
