graph TD
    %% 1. Defined first to stay at the top
    subgraph Strategic_Alignment ["Strategic & Performance Goals"]
        direction TB
        BV1["Hospital Resource Management"]
        BV2["Patient Satisfaction through infromed Shared Decision"]
        
        CompanyKPI1["Cost Effective"]
        CompanyKPI2["Earlier Access to Appropriate Care"]
        CompanyKPI3["Optimised Resource Allocation"]

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
        Y["Y Variable (Target):</b><br/>Probability of failing to reach<br/>Minimal Important Difference (MID)"]
    end

    %% 3. Arrow from Top to Bottom using the 'backwards' syntax
    Strategic_Alignment -- "Explain Patient why surgery will fail" --> Model_Variables
