# 統合レビュー: Minimal ALife Project Overview

> **Document role**: Research review — critiques and recommendations against the initial proposition.
>
> **Input**: [`minimal_alife_project_overview.md`](./minimal_alife_project_overview.md) (the initial proposition)
> **Output**: [`technical_design_spec.md`](./technical_design_spec.md) (design decisions resolving the issues raised here)
>
> Two independent reviews were conducted and merged into this unified analysis.
> Duplicates removed, complementary points consolidated, and prioritized.

---

## A. 根本的な問題（研究の成立に関わる）

### A1. 研究上の問い（Research Question）が不在

プロジェクトの「何を明らかにするか」が宣言されていない。「進化的ダイナミクスが創発する世界を作る」はビジョンであり、検証可能な主張ではない。以下のいずれかを立てるべき：

- コード生成タスクにおいて進化的手法はRL/勾配ベースと何が質的に異なるか
- LLMを変異オペレータとして使うことでGP（遺伝的プログラミング）を超える複雑性が生まれるか
- 最小エネルギーモデルでニッチ分化やアームズレースは創発しうるか

### A2. 「ALife創発」の判定基準が定性的で反証不能

Section 15の「生命らしさの兆候」は全て定性的記述。研究として成立させるには：

- **事前登録した定量KPI**（系統多様性指数、適応速度、戦略クラスタ数など）
- **統計検定と対照条件**（ランダム変異ベースライン vs 提案手法）
- **反証条件の明示**（何が観測されなければ「創発しなかった」と言えるか）

### A3. 先行研究との位置づけが完全に欠落

Tierra, Avida, Lenia, OpenAI Neural MMO, 遺伝的プログラミング（Koza）、LLM-based code evolution（AlphaCode, FunSearchなど）との差分が一切記述されていない。既知の知見の上に何を積むのかが不明。

---

## B. 設計上の矛盾・欠陥

### B1. 「客観・決定的」の主張と実行時間スコアの矛盾

Section 6で「Same input → same score」を要件としながら、評価関数に `time_seconds` を含めている（Section 10）。実行時間はOS負荷・キャッシュ・GCで揺らぐため、決定論性を満たさない。

**対策案**: 命令数カウント、メモリアクセス回数、Big-O推定などの決定的な計算量指標に置換する。あるいは複数回実行の中央値＋信頼区間で安定化する。

### B2. Goodhart化（テスト最適化ハック）への防御がゼロ

評価が「テスト通過率＋時間＋編集コスト」のみの場合、エージェントはテストケースのハードコーディング、テスト入力のパターンマッチ的回避、テスト順序への依存といった「スコア最大化の抜け道」を学習しうる。

**対策案**: テストセットをtrain/validation/hiddenに分離し、hiddenテストでの汎化性能を真のfitnessとする。あるいはproperty-based testing（Hypothesis等）で入力空間を動的に拡張する。

### B3. 進化 vs 最適化の区別がついていない

単一エージェントの自己変異ループ（Section 13）は、遺伝（世代間情報伝達）を欠いているため、進化ではなくヒルクライミング。`fitness > previous_fitness` 中心の受容は早期収束しやすく、多様性維持戦略（Novelty Search、entropy正則化、MAP-Elitesなど）がない。

**対策案**: 複数エージェントの集団を維持し、交叉・選択・世代交代を導入する。あるいは単一エージェントでも、品質多様性（Quality-Diversity）アルゴリズムを参照する。

---

## C. 未定義の核心設計

### C1. 変異メカニズムの主体と方法

「誰が」変異を生成するかが書かれていない。LLMがプロンプトで生成するのか、AST変換か、ランダムな文字列操作か。ここがプロジェクトの最重要設計であり、最も詳細な記述が必要。

### C2. 確率的受容の設計

`reject_or_probabilistic_accept()` の受容確率関数が未定義。温度パラメータの有無、固定か適応的か、シミュレーテッド・アニーリング的なスケジュールかが不明。

### C3. 変異コード実行の安全設計

タイムアウトの記述はあるが、I/O制限、syscall制限、ネットワーク遮断、ファイルシステム隔離などのサンドボックス設計が欠如。自律エージェントが生成するコードを無制限に実行するのは重大なリスク。

### C4. 重要パラメータの根拠

| パラメータ | 問題 |
|-----------|------|
| `0.8 / 0.15 / 0.05` の重み | 根拠なし。感度分析の計画もない |
| `N steps`（改善なしでの死亡閾値） | 未定義 |
| `edit_cost` の定義 | Levenshtein距離？AST差分？行数？ |
| 初期エネルギー `1.0` | スケールの根拠なし |

### C5. 再現性要件

乱数シード固定、実験回数、分散報告、ハードウェア条件が明記されておらず、第三者による結果比較が不可能。

---

## D. 主張の過大さ

### D1. 「人間を完全にループ外へ」は過大主張

評価関数の設計自体が強い人間バイアス。重み選択、タスク選択、死亡条件の設定はすべて人間の価値判断。「実行ループから人間を排除」は正確だが、「完全自律」は過大。

### D2. Section 2のパラダイム議論は本題と乖離

「Ideas are cheap, execution is cheaper」は面白い観察だが、ALife設計との接続が弱い。論文化するなら簡潔な動機づけに圧縮すべき。

---

## E. 統合推奨アクション（優先順位付き）

| 優先度 | アクション | 対応セクション |
|--------|-----------|---------------|
| **Critical** | Research Questionを1〜2個、反証可能な形で明示する | 新規追加 |
| **Critical** | 「ALife創発」の定量判定基準を事前定義する（多様性指数、系統持続性、適応速度など） | Section 15 |
| **Critical** | 変異メカニズムの具体設計を記述する（主体・方法・プロンプト設計） | Section 12 |
| **High** | 先行研究（Tierra, Avida, GP, FunSearch等）との差分を記述する | 新規追加 |
| **High** | 実行時間を決定的な計算量指標に置換する、または安定化手法を導入する | Section 10 |
| **High** | テストセット分離（train/validation/hidden）でGoodhart化を防ぐ | Section 10 |
| **High** | サンドボックス実行基盤の設計を追加する | Section 7, 13 |
| **Medium** | 単一エージェント最適化と集団進化の区別を明確化し、多様性維持戦略を検討する | Section 13 |
| **Medium** | 全パラメータの根拠と感度分析計画を記述する | Section 10, 11 |
| **Medium** | 再現性要件（シード、試行数、分散、HW条件）を明記する | 新規追加 |
| **Medium** | 「完全自律」の主張を「実行ループからの人間排除」に限定する | Section 2, 16 |
| **Low** | Section 2のパラダイム議論を簡潔化または分離する | Section 2 |

---

## F. 方向性の確認（回答が必要な問い）

1. **研究の性格**: 工学デモ（「動くものを作る」）か、学術主張（「Xが創発することを示す」）か？後者なら上記Critical項目はすべて必須
2. **「生命らしさ」の定量指標**: 多様性（Shannon index）、系統持続（lineage depth）、適応速度（fitness curve傾き）のうち、どれを主要KPIとするか？
3. **汎化テスト**: テストをtrain/validation/hiddenに分離する前提で進めるか？
4. **変異の主体**: LLMベースの変異を想定しているか？その場合、LLM呼び出しコストをエネルギーモデルに組み込むか？
