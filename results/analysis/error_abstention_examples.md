# Ten illustrative LODO examples

Fixed threshold 0.4; deterministic diagnostic selection, not representative sampling. Label 0 is Satire for FakeVsSatire. Excerpts below are limited to 500 characters; model input was max 128 tokens.

## 1. ISOT — ISOT:Fake.csv:14213

True binary label: 1; prediction: 0; compatibility: 0.5393; abstained: False. Selection: highest-compatibility error.

> According to a Gallup poll released on Thursday, Republican satisfaction with the direction in which the USA is heading took its biggest hit between the months of May and June in the time since Donald Trump took office, dropping a massive 17 points.While satisfaction among Democrats and independents also fell during the same time period, one and three points, respectively, the level of satisfaction among Republicans plummeted from 58% in May to only 41% in June. To add insult to injury, these la

## 2. ISOT — ISOT:True.csv:7883

True binary label: 0; prediction: 1; compatibility: 0.0000; abstained: True. Selection: lowest-compatibility abstention.

> U.S. regulations, proposed by the Treasury to crack down on companies that try to reduce taxes by rebasing abroad, have begun a White House review and could be finalized shortly, officials said on Tuesday. The regulations, which would make it difficult for U.S. business operations to avoid taxation while shifting profits overseas through a practice called “earnings stripping,” were received by the White House Office of Management and Budget (OMB) last week. The agency has up to 90 days to decide

## 3. ISOT — ISOT:True.csv:3308

True binary label: 0; prediction: 0; compatibility: 0.0000; abstained: True. Selection: lowest-compatibility correct prediction.

> Illinois must increase payments to Medicaid providers despite an ongoing budget impasse, after a U.S. District Court judge on Wednesday ruled the minimal payments made by the state do not comply with federal consent decrees. Judge Joan Lefkow ordered renewed negotiations between Illinois and health care advocates for the poor, setting a June 20 deadline to be in “substantial compliance” with the decrees. Lawyers representing the state’s 3 million Medicaid recipients had asked the judge to give p

## 4. LIAR_PLUS — LIAR_PLUS:6720.json

True binary label: 1; prediction: 0; compatibility: 0.9969; abstained: False. Selection: highest-compatibility error.

> U.S. House opponent Paul Ryan supports cutting funding for veterans by 24 percent.

## 5. LIAR_PLUS — LIAR_PLUS:10223.json

True binary label: 1; prediction: 0; compatibility: 0.0009; abstained: True. Selection: lowest-compatibility abstention.

> The State Department says the 42,000 jobs created by the Keystone XL pipeline are ongoing, enduring jobs.

## 6. LIAR_PLUS — LIAR_PLUS:10271.json

True binary label: 0; prediction: 0; compatibility: 0.0052; abstained: True. Selection: lowest-compatibility correct prediction.

> Atlanta Mayor Kasim Reed became a city resident in the last annexation, opening the door to his candidacy.

## 7. FakeVsSatire — FakeVsSatire:Fake/finalFake:395.txt

True binary label: 1; prediction: 0; compatibility: 0.9895; abstained: False. Selection: highest-compatibility error.

> MEXICAN PRESIDENT: “WE AGREE TO PAY FOR THE WALL IF THE DEAL INCLUDES RETURNING TEXAS TO MEXICO” U.S. President Donald Trump said he had a friendly phone call with Mexico’s president on Friday but asserted he will renegotiate trade deals and other aspects of the countries’ ties because Mexico has “beat us to a pulp” in the past. Mexico’s peso extended gains on news of the call. The two men spoke for about an hour on Friday, a day after President Enrique Pena Nieto scrapped a planned meeting in W

## 8. FakeVsSatire — FakeVsSatire:Satire/finalSatire:595.txt

True binary label: 0; prediction: 0; compatibility: 0.4107; abstained: False. Selection: lowest-compatibility correct prediction.

> Breaking: Conservative Firecracker Tomi Lahren Just Beat The Crap Out Of Sniveling Miley Cyrus Tomi Lahren is well known to be a bit short-tempered sometimes, making sure everyone knows that her petite size and adorable appearance doesn’t mean she’s some kind of a pushover. As with most strong conservative women, not only does she shoot with deadly accuracy, she also makes a mean sandwich. Liberals, for some reason, think sandwich making makes a woman less of a person, yet men can make all the s

## 9. ISOT — ISOT:True.csv:8417

True binary label: 0; prediction: 1; compatibility: 0.0000; abstained: True. Selection: additional uncertainty example.

> Senator Ben Sasse was in a fairly lonely place when he wrote on his Facebook page back in February that he could not support Republican presidential candidate Donald Trump. That is changing. Since Sasse, a Nebraska Republican, staked out his position, the ranks of Republican Trump critics in the U.S. Congress have grown, with the New York real estate mogul stirring almost constant turmoil within his party. The following are Republicans in Congress who have said they will not vote for Trump in th

## 10. LIAR_PLUS — LIAR_PLUS:1950.json

True binary label: 1; prediction: 0; compatibility: 0.0015; abstained: True. Selection: additional uncertainty example.

> Four years as a congressman, he [U.S. Rep. Hank Johnson] never talked about MARTA.
