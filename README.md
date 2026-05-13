# EOM PoC v2 — 組織ファジングによる統制破綻の探索

## 概要

現実に近い社内規程（購買管理規程・経理処理基準・承認権限表・ワークフロー運用規程等）と業務データを準備した環境に、自律的なLLMエージェントを「1人の社員」として配置し、業務を遂行させる。その過程を反復観察することで、**どの条件で統制破綻が再現するか**を測定する。

2026年5月の再設計では、当初の「モデリングしない」という主張を撤回し、役割、権限、外部圧力、業務負荷、統制強制レベルを明示的な実験因子として扱う。規程テキストを形式ルールへ完全変換するのではなく、自然言語規程を読んだエージェントがどのように解釈し、どこで調整不全や統制逸脱を起こすかを測る。

詳細な批判的レビューと再設計は [docs/05_research_redesign.md](docs/05_research_redesign.md) を参照。

## 現在の研究方針

本PoCは「AIエージェントによる組織ファジング」と位置づける。

- 規程テキスト、案件データ、担当者構成、外部圧力を入力として与える
- 同一条件を複数回実行し、破綻パターンの再現率を測る
- soft / monitored / hard の統制強制レベルを分ける
- 二重支払、自己承認、ワークフローバイパス、履歴欠落、金額driftなどを機械採点する
- 発見を「統制破綻」「解釈差」「調整不全」「ハーネス由来」「モデル能力由来」に分類する

## 先行研究 EOM v1 からの経緯

本PoCは、Executable Organizational Model (EOM) v1（[digital_twin リポジトリ](https://github.com/yujihama/digital_twin), PR #1〜#30）の知見を踏まえた次世代の設計である。

### v1 で見えたこと

v1では合成環境（environment.py / rules.py）の中で購買承認フローをシミュレートし、約200セルのLLM実験を実行した。主な発見:

1. **Quiet drift**: LLMエージェント（vendor_e）は明示的な「不正アクション」を選ぶのではなく、通常のdeliver/register_invoiceで金額を±2.5%ずらす「解釈の揺れ」を示した（T-028, 25%の再現率）
2. **複合曖昧さ効果**: 単一の曖昧さ（税込/税抜不明、前回調整、概数発注）では揺れが発生しないが、複数の曖昧さが同時に存在すると金額解釈の自由度が生まれる
3. **モデル固有性**: gpt-4.1-miniで25%のdrift rateを示した条件で、Claude Sonnetは0%。driftは環境の構造的帰結ではなくLLMの推論パターンに依存
4. **明示的ルール vs 暗黙的ルール**: buyer_aはペルソナに明示された規則（分割発注禁止）を完全遵守。vendor_eは暗黙的な期待（PO通りの金額）に対してのみ揺れる
5. **tolerance閾値**: 三者照合のtolerance 3%で全driftが吸収され、組織のサーベイランス信号が消失

### v1 の限界と v2 の動機

v1の知見は全て「研究者がenvironment.py / rules.pyを書いた合成環境」の中で得られた。以下の限界が認識されている:

- **モデリングの恣意性**: 環境変数（approval_threshold, vendor_profit_margin等）の選択自体が因果仮定を含む
- **action spaceの事前定義**: エージェントの行動をActionOptionとして列挙すること自体が、「何ができるか」を研究者が決定している
- **規程のハードコード**: 「承認閾値20万円」がPythonのif文で実装されており、現実の規程テキストの曖昧さが排除されている
- **単発LLM call**: エージェントが自律的に情報を探索する能力がない

当初のv2はこれらの限界を、**モデリング層を排除する**ことで解消しようとした。しかし、現在の批判的レビューでは、この方針は研究設計として不十分と判断している。実際には、エージェント構成、権限、ワークフロー、入力データ、実行順序を研究者が設計しているためである。

再設計後は、モデリングを排除したと主張するのではなく、モデル化している部分を実験因子として明示し、条件を変えたときの統制破綻率を測る。

## 設計思想

### モデリングを明示する

v1:
```
研究者がenvironment.pyを書く → 研究者がrules.pyを書く → LLMがその中で行動する
```

当初v2:
```
現実の規程テキストをファイルとして置く → 現実の取引データをファイルとして置く
→ 自律エージェントが自分で読み、自分で解釈し、自分で行動する
```

再設計後:
```
規程・案件・役割・権限・外部圧力・統制強制レベルをmanifestで固定する
→ 同一条件を反復実行する
→ 不変条件チェッカーで破綻イベントを採点する
→ 条件差による破綻率の変化を見る
```

研究者は何をモデル化したかを明示する。主張の対象は「モデル化を排除した現実再現」ではなく、「条件を制御したサンドボックス内での統制破綻の再現性」である。

### エージェントは「1人の社員」

各エージェントに与えるのは:
- **役割**（購買担当、承認者、経理担当、取引先）
- **能力**（ファイル読み書き、検索、メッセージ送信）
- **業務の入口**（inboxの業務依頼）

具体的な業務手順は与えない。エージェントが規程を読んで自分で判断する。

### 揺れと破綻の検出はイベントから

同じ入力に対して、複数回の実行 / 複数モデルでの実行を比較し、行動の分散が大きい箇所を「解釈が揺れている箇所」として扱う。ただし、研究成果としてはそれだけでは不十分である。

再設計後は、二重支払、越権承認、ステータス順序違反、history欠落、current_handler無視、金額差異、JSON破損などをイベントとして機械採点し、解釈差・調整不全・統制破綻・モデルアーティファクトを分離する。

## 技術スタック

- **エージェントフレームワーク**: [LangChain DeepAgents](https://docs.langchain.com/oss/python/deepagents/overview)
  - 組み込みファイルシステム操作（ls, read_file, write_file, edit_file）
  - タスク計画（write_todos）
  - サブエージェント生成
  - コンテキスト管理（auto-summarization）
  - 権限制御（permissions）
- **トレーシング**: LangSmith
- **LLMモデル**: gpt-4.1-mini, Claude Sonnet（マルチモデル比較用）

## ディレクトリ構成

```
Agent-Audit-Sandbox/
├── README.md                    # 本ファイル
├── docs/
│   ├── 01_background.md         # 経緯と研究的位置づけ
│   ├── 02_design.md             # 設計詳細
│   ├── 03_analysis_plan.md      # 分析計画
│   ├── 04_run_findings.md       # 既存実行の観察メモ
│   └── 05_research_redesign.md  # 批判的レビューと再設計
├── workspace/                   # エージェントの作業環境
│   ├── regulations/             # 規程類（読み取り専用）
│   ├── transactions/
│   │   └── pending/             # 業務依頼（初期配置）
│   ├── shared/                  # エージェント間共有
│   │   ├── purchase_requests/
│   │   ├── approved/
│   │   ├── orders/
│   │   ├── deliveries/
│   │   ├── invoices/
│   │   ├── payments/
│   │   └── messages/
│   ├── vendor_context/          # 取引先情報（vendor_agentのみ）
│   ├── outbox/                  # 最終出力
│   └── logs/                    # 実行ログ
├── agents.py                    # エージェント定義
├── runner.py                    # 実行ループ
├── requirements.txt
└── .env.example
```

## 始め方

```bash
pip install deepagents langchain-openai langchain-anthropic
cp .env.example .env
# .env に OPENAI_API_KEY, ANTHROPIC_API_KEY, LANGSMITH_API_KEY を設定
# workspace/regulations/ に規程テキストを配置
# workspace/transactions/pending/ に業務依頼を配置
python runner.py
```

## 関連リポジトリ

- [digital_twin](https://github.com/yujihama/digital_twin) — EOM v1（PR #1〜#30: 合成環境での統制感度分析）
