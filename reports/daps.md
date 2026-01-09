# DAPS Diagram for EAISI NHS PROMS Project

## Business Value
Our main goal is to reduce failure rates post-surgery after 1 year by predicting which patients will have poor outcomes from hip and knee replacement surgeries. This enables:
- Proactive identification of high-risk patients before surgery
- Informed clinical decision-making to potentially avoid unnecessary procedures
- Improved patient outcomes and satisfaction through better preoperative counseling
- Significant cost savings for the healthcare system by reducing failed surgeries and associated complications

```mermaid
graph TD
    %% 1. Defined first to stay at the top
    subgraph Strategic_Alignment ["Strategic & Performance Goals"]
        direction TB
        BV1["<b>Hospital:</b> Enhanced Operational Efficiency<br/>via targeted surgical referrals"]
        BV2["<b>Patient:</b> Satisfaction through informed Shared Decision"]
        
        CompanyKPI1["Cost Effective"]
        CompanyKPI2["Earlier Access to Appropriate Care"]
        CompanyKPI3["Optimised Clinical<br/>Resource Allocation"]

        ProjectKPI1["Reduction in failed surgery"]
        
        BV1 --> CompanyKPI3
        
        BV1 --> CompanyKPI1
        BV2 --> CompanyKPI2
        BV2 --> CompanyKPI1
        CompanyKPI2 --> ProjectKPI1
        CompanyKPI1 --> ProjectKPI1
        CompanyKPI3 --> ProjectKPI1

        
    end

    %% 2. Defined second to stay at the bottom
    subgraph Model_Variables ["Data & Outcome Variables"]
        direction LR
        X["X Variables (Features)<br/>Age, BMI, Comorbidities,<br/>Baseline Scores"]
        Y["<b>Y Variable (Target):</b><br/>Probability of failing to reach<br/>Minimal Important Difference (MID)"]
    end

    %% 3. Arrow from Top to Bottom using the 'backwards' syntax
    Strategic_Alignment -- "Explain Patient why surgery will fail" --> Model_Variables
```



