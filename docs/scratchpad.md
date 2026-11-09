# RZ.Foundation.AI scratchpad

## 2026-11-08 — Cost model and calculation review

The RA-311 library implementation is in commit `3716129`. The enduring description is in
[the library design](design.adoc). This scratchpad retains the rationale and verification history.

The user accepted `TokenCost` as the name for the tokens-and-USD pair and preferred its descriptive name to
the shorter `Cost`. The chat amount has Input, Output, and Other, each a `TokenCost`. Thought usage and its
amount belong to Other. The design separates prices into `ChatRate`.

The existing `ChatCost` serves both as a USD-per-million rate in `CostStructure` and as an incurred satang
amount. The revision separates those meanings and retains the provider usage that is currently dropped.

### Actual pricing check

The user asked to verify whether the calculation should remain the same. An offline .NET console harness
referenced this repository's actual library project and called `LLM.CalcCost`. It made no provider calls.
Source revision: `99a08711cedb6702292c7319ed6f45245aeef16d`.

Expected values came from the configured sample rates and, for the GPT-5.4 long-context condition, the official
[GPT-5.4 model documentation](https://developers.openai.com/api/docs/models/gpt-5.4). The official
[GPT-5.5 documentation](https://developers.openai.com/api/docs/models/gpt-5.5) describes the same input threshold.
Actual satang amounts were divided by 3800 for comparison in USD.

| Case | Expected USD input / output / total | Actual USD input / output / total | Result |
| --- | --- | --- | --- |
| Standard: 1,000 input, 200 output; rates 2.5 / 15 | 0.0025 / 0.003 / 0.0055 | 0.0025 / 0.003 / 0.0055 | Matched |
| Thought pricing: 1,000 input, 200 response, 300 thought; rates 0.15 / 0.6 / 3.5 | 0.00015 / 0.00117 / 0.00132 | 0.00015 / 0.00117 / 0.00132 | Matched |
| GPT-5.4: 300,000 input, 1,000 output | 1.5 / 0.0225 / 1.5225 | 1.5 / 0.015 / 1.515 | Mismatch |
| GPT-5.4: 272,000 input, 1,000 output | 0.68 / 0.015 / 0.695 | 1.36 / 0.015 / 1.375 | Mismatch |

`LLM.cs:23` chooses the output rate using the output token count. For these long-context model rules, both
rates are selected by input count. `LLM.cs:21` also uses `>=` where the documented condition is above 272K.
Filed [RA-315](https://pace6.atlassian.net/browse/RA-315) under epic
[RA-314](https://pace6.atlassian.net/browse/RA-314). GPT-5.4's two failures were reproduced; GPT-5.5 was not
separately exercised. Full repository tests and live billing comparisons were not run.

Retain the usage-times-rate arithmetic and remove the currency conversion, while treating the verified tier
selection defects as corrections. Preserve the pricing capabilities without copying those defects into the new design.

### Pricing policy decided during the interview

The user explicitly chose to allow models whose prices are absent from the library's built-in table and
confirmed that cost = 0 is acceptable. Keep the existing zero-rate fallback unless the caller supplies rates.
Do not introduce an unknown-cost state. Retain token counts when the monetary estimate is zero.

The behavior and the explicit-rate option are documented in the existing `README.adoc`. The library design
records the same policy.

### RA-315 scope decision

The user approved including RA-315's tier-selection and 272K boundary corrections in the RA-311 library
change, with focused regression tests. RA-315 remains separately tracked under cost-correctness epic RA-314.
This does not decide the scope of the other cost bugs.

### RA-312 scope decision

The user explicitly declined including RA-312 in the RA-311 library change. Keep its fix as a separate
follow-up under RA-314. The RA-311 library work retains the existing failure-result contract; RA-315's
calculator corrections remain included.

### Input, Output, and Other

The user selected three generic cost buckets: Input, Output, and Other. Thought tokens and their monetary
amount belong to Other. Other also provides a place for future charges outside input/output without adding
model-specific cost structures. Each bucket retains tokens plus USD; totals sum the three disjoint buckets.

### Snapshot semantics

The user clarified that token counts and USD amounts are snapshots supplied by the model. The responsibility
is to track them as accurately as practical, without asking how a particular combination arose or adding
cross-field rules to explain it. Preserve both supplied values, their attribution, and their aggregates.
Calculation of an estimate belongs to snapshot production; tracking a captured snapshot does not reprice it.

### Target class diagram

The user requested a target library class diagram in the task design. It shows the agreed snapshot model
and the existing integration points, with a concrete pricing-policy structure labelled as a proposal.
FlatCostStructure and TieredCostStructure represent the two pricing forms already observed in the rate
tables. The proposal keeps the threshold and second rate together in the tiered form rather than filling
unused threshold/rate fields in the flat form. Other pricing covers thought usage without another policy type.

The user explicitly instructed that diagram renderability must not be verified. No renderer or diagram
syntax/render validation was run; the user will report any rendering errors.

### Implementation completed in working tree

The user authorized implementation of the library design and instructed that commit/push remain their
responsibility. No commit or push was performed.

Implemented TokenCost/ChatCost snapshots, separate ChatRate and flat/tiered policies, USD calculations,
OpenAI reasoning-to-Other mapping, and RA-315 corrections. Simple pricing defaults Other to the output
rate to preserve the price of reasoning included in OpenAI output; callers can supply a different third
rate. Gemini retains its distinct thought price and existing first-entry attribution.

Verification: all 76 offline unit tests passed. The complete Release solution built successfully.
FoundationAITests passed 7 tests and skipped its 5 live-provider tests via the existing RunTests=false
configuration. git diff --check passed. No provider API requests or diagram render/syntax checks were run.

Source inspection also found completed provider usage lost on response-conversion errors. Filed
[RA-316](https://pace6.atlassian.net/browse/RA-316) under RA-314, separately from the deferred RA-312
resolver issue. This is source-level evidence, without a runtime reproduction of the failure cases.

## 2026-11-09 — Cached input, validation, and closed policies

The user approved using reported cached-input usage when producing cost snapshots, with no new snapshot
bucket or retained cached-token breakdown. Input.Tokens stays the full count; Input.Usd combines ordinary
and cached prices. Explicit caller rates can supply a cached price; omission uses the ordinary input price.
Built-in OpenAI cached prices were checked against official pricing on this date, including the higher
cached prices above the long-input threshold.

Sources:
- https://developers.openai.com/api/docs/pricing
- https://developers.openai.com/api/docs/models/gpt-4.1-nano
- https://developers.openai.com/api/docs/models/gpt-4o
- https://developers.openai.com/api/docs/models/gpt-5
- https://developers.openai.com/api/docs/guides/prompt-caching
- Installed OpenAI 2.12.0 XML documentation: ChatInputTokenUsageDetails.CachedTokenCount.

The user approved constructor validation for nonnegative snapshots and rates. Properties are get-only
to prevent with/init bypasses; JSON uses the checked constructors. Zero-token/positive-USD and
positive-token/zero-USD snapshots remain valid. The calculator validates the usage arguments it subtracts.

The user chose a closed policy family: flat and tiered; callers supply rates. The base and concrete
policies use classes with a private protected base constructor and sealed concrete classes. Records
were avoided here because their protected copy-constructor surface would allow another derivation path.
The explicit ChatRate constructor requires the Other price, removing the misleading zero default noted
in Claude's review item 2.

The updated offline unit suite passed all 82 tests. New coverage includes no/partial/full cache hits,
every existing OpenAI model's cached price, exact and above-threshold cached rates, ordinary-price fallback,
explicit zero and unknown-model pricing, negative construction/deserialization, get-only properties,
JSON round-trip, and invalid usage subsets. Release solution build passed. FoundationAITests passed 7 tests and skipped its 5 live-provider cases
under the existing RunTests=false configuration. The code diff whitespace check passed.

Verification/history sections previously in the task design are already recorded in this scratchpad;
the design now records the decisions and scope instead. No diagram renderer, syntax checker or document
validation was run. No commit, push, or staging action was performed.

### ChatRate JSON fallback re-review

Astra's omitted-price case reproduced offline: input priced at USD 2 per million, 10,000 input tokens,
8,000 cached, no cached price in the JSON. The original JSON constructor bound the missing decimal to
zero and produced USD 0.004 rather than USD 0.020.

The JSON boundary now distinguishes omission from an explicit zero and constructs ChatRate through its
existing validation. An explicit zero is written even when the serializer omits other default values.
The regression covers omitted, explicit zero, and explicit nonzero prices, plus a zero-price JSON
round trip with default-value omission and camel-case naming.

After the fix all 86 offline unit tests passed, the Release solution built, and the source/test diff
whitespace check passed. No commit or push was performed.

## 2026-11-09 — Library design handoff

The durable design covers the chat contract, provider adapters, tool resolver, cost snapshots, pricing,
and JSON. The completed library check was a Release build with 96 unit tests and 7 Foundation tests
passing. The 5 live-provider tests were skipped by their existing configuration.

Gemini passes reported cached input to the calculator. Its built-in model and price refresh is a
follow-up. [RA-317](https://pace6.atlassian.net/browse/RA-317) records the pre-existing GPT-5.1
pricing lookup typo.
