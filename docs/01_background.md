# 01. 経緯と研究的位置づけ

## 1. EOM v1 の実験系列

EOM v1（[digital_twin](https://github.com/yujihama/digital_twin), 2026年4月）では、購買承認フローの合成環境を構築し、LLMマルチエージェントシミュレーションにより統制の感度を分析した。PR #1〜#30、約200セルのLLM実行を通じて以下の知見を得た。

### Phase 1: フレームワーク構築と基礎実験（PR #1〜#20）

- 5体のエージェント（buyer_a, buyer_b, approver_c, accountant_d, vendor_e）による購買承認フローの実装
- environment.py / rules.py による決定論的な環境ルール
- exp001〜exp005: 需要生成、介入実験（承認閾値変更、三者照合無効化）
- Layer 1-3 検証プロトコル: 創発性（Emergence Ratio 62.5%）、経路依存性、相互作用遮断

### Phase 2: 研究方向性の転換（PR #21〜#22）

外部ヒアリング3回を経て、研究の軸を以下のように再定義した。

| 項目 | 旧 | 新 |
|------|-----|-----|
| 中心主張 | DAG-freeな因果推論 | intuition-failure frontierの発見 |
| システム名 | Organizational Causal Twin (OCT) | Executable Organizational Model (EOM) |
| LLMの位置づけ | 環境シミュレータの核 | 置換可能なpolicy module |
| 主要指標 | Emergence Ratio | Query-Simulation Divergence (QSD) |

### Phase 3: Ablation と deviation frontier（PR #23〜#30）

- **T-021（Baseline Ladder）**: RB-min（L1）vs LLM（L3）の比較。L3はL1より遅い（baseline payments: L1=21.67 > L3=18.67）が、介入への応答が大きい（I1 ATE: L3=+2.33 vs L1=+0.67）
- **T-022（vendor incentive）**: observation-levelの数値incentiveでは全30セルdeviation=0。LLMのコンプライアンスバイアスは数値信号だけでは揺らがない
- **T-023（narrative framing）**: business_contextをテキスト化した場合、L3_high_pressure seed43でdeliver_partial（646k円を半量納品）が1件出現。5,100+の決定機会で初のdeviation
- **T-028（解釈の曖昧さ）**: 注文にtax_included/prior_adjustment/quantity_specの曖昧さを導入。通常のdeliver/register_invoiceで金額が±2.5%ずれるquiet driftを観測。25%の再現率（5/20セル）。tolerance 3%で全吸収
- **T-028 follow-up**: 複合曖昧さ効果（単一ブランチでは不発、全チャネル同時でのみ出現）、narrative × ambiguity非干渉（dominant channel型）
- **T-029（多方向探索）**: temperature非単調（0.5-1.2で0-20%）、buyer_a分割発注0/20（persona遵守）、Claude Sonnet 6/6でdeviation=0（モデル固有）

## 2. v1 の根本的な限界

### モデリングの恣意性

v1の全ての知見は「研究者がenvironment.py / rules.pyを書いた合成環境」の中で得られた。環境変数の選択（approval_threshold, vendor_profit_margin, three_way_match_tolerance等）、action spaceの定義（deliver, deliver_partial, invoice_with_markup等）、曖昧さフィールドの設計（tax_included, prior_adjustment, quantity_spec）は全て研究者の因果仮定を含む。

この問題は研究の初期段階からYujiにより繰り返し指摘された:

> 「環境条件の軸すらもあらかじめ決めるのは恣意的である」

外部ヒアリングでは「完全には解けない。limitationとして記述する」（選択肢A）が採用されたが、v2ではこの問題に正面から向き合う。

### 「不正の意思決定」vs「解釈の揺れ」

v1のT-022/T-023は「LLMがopportunistic actionを選ぶかどうか」というbinary decisionとしてdeviationを扱った。しかしT-028で見えたのは、不正は「意思決定」ではなく「解釈の揺れの蓄積」として出現するということだった。

この知見はYujiの実務経験と一致する:

> 「人間の場合はコンプライアンスが低いのではなく、解釈が揺れることで、ケースによっては不正ととらえられるケースが多い」

v2はこの構造を、合成的な曖昧さフィールドではなく、現実の規程テキストの曖昧さから自然に再現することを目指す。

### 単発LLM callの限界

v1のエージェントはobservationを受け取り、1回のLLM callでJSON応答を返すだけ。記憶なし、自律的情報探索なし。現実の社員は規程を参照し、過去の経験を想起し、上司に相談する。この過程自体が統制の機能/形骸化に影響する。

## 3. v2 の位置づけ

v2はv1の延長ではなく、**設計思想の転換**。

| | v1 | v2 |
|---|---|---|
| 環境 | 研究者がコードで定義 | 現実の規程・データをファイルとして配置 |
| ルール | rules.pyにハードコード | 規程テキストをエージェントが解釈 |
| 行動空間 | ActionOptionとして列挙 | ファイル操作 + メッセージ送信（制約なし） |
| エージェント | 単発LLM call | 自律エージェント（計画、記憶、情報探索） |
| deviation検出 | three_way_match等の決定論的ルール | 複数実行間の行動分散 |
| 恣意性 | 環境変数の選択に内在 | 入力データの選択に限定 |

v1の知見（quiet drift、複合曖昧さ効果、モデル固有性）はv2で検証仮説として活きる:

- quiet driftは現実の規程でも再現されるか
- 複合曖昧さ効果は規程テキストの自然な曖昧さでも成立するか
- モデル間差異は現実データでも同方向か
- 明示的ルール vs 暗黙的ルールの境界は現実ではどこにあるか
