# Missing Values Handling Decisions

This document outlines the decisions made regarding the handling of missing values in the dataset during the data preparation phase of the NHS PROMs project. These decisions follow the CRISP-DM methodology, focusing on data cleaning and imputation to ensure data quality for modeling.

## Overview
Missing values can significantly impact model performance and lead to biased results. Decisions were based on:
- Domain knowledge (e.g., medical context for PROMs data).
- Data exploration (e.g., assessing patterns of missingness).
- Best practices for imputation to avoid data leakage and maintain generalizability.

## Decisions by Column/Category

### Indicator Columns (Disease/Condition Flags)
For binary indicator columns representing the presence/absence of conditions (e.g., Arthritis, Cancer, Circulation, Depression, Diabetes, Heart Disease, High Bp, Kidney Disease, Liver Disease, Lung Disease, Nervous System, Stroke), missing values were replaced with `0` (indicating absence of the condition).

- **Rationale**: In medical datasets, missing indicators are often interpreted as "not present" or "not reported," which aligns with assuming absence to avoid overestimating prevalence. This is a conservative approach to prevent false positives in health-related predictions.
- **Implementation**: Used pandas `fillna(0)` on the specified columns.
- **Impact**: Reduces missingness without introducing bias, as these are categorical flags.

### Gender Column
- **Missing Values Count**: 8686
- **Decision**: Pending further review. Options include:
  - Imputation based on mode (most frequent value).
  - Dropping rows if missingness is not random.
  - Advanced imputation (e.g., using other features like age or provider data).
- **Rationale**: Gender is a key demographic variable. Missing values may indicate data entry issues or privacy concerns. Further analysis needed to assess if missingness is Missing Completely at Random (MCAR), Missing at Random (MAR), or Not Missing at Random (NMAR).
- **Next Steps**: Conduct pattern analysis (e.g., correlate with other variables) before final imputation.

### Pre-Op Q EQ VAS (Pre-Operative EQ-VAS Score)
- **Missing Values Count**: 6372
- **Decision**: Impute using median or mean of the column, or advanced methods like KNN imputation.
- **Rationale**: EQ-VAS is a continuous score measuring health-related quality of life. Missing values could be due to incomplete surveys. Imputation preserves sample size for modeling.
- **Implementation**: Considered in iterative data preparation; avoid simple mean if outliers are present.

### Post-Op Q EQ VAS (Post-Operative EQ-VAS Score)
- **Missing Values Count**: 3435
- **Decision**: Similar to Pre-Op Q EQ VAS; impute using statistical measures or predictive imputation.
- **Rationale**: This is a key outcome-related variable. Missing post-op scores may correlate with poor outcomes or incomplete follow-up. Imputation ensures continuity for longitudinal analysis.
- **Implementation**: Use domain-informed imputation (e.g., based on pre-op scores or clinical knowledge).

## General Handling Strategy
- **Initial Approach**: In early iterations, rows with any missing values were dropped to assess baseline data quality (as per CRISP-DM Data Understanding phase).
- **Iterative Refinement**: Moved to imputation for production-ready models, prioritizing methods that prevent data leakage (e.g., using training set statistics only).
- **Tools Used**: Pandas for basic imputation; scikit-learn pipelines for scalable, cross-validation-safe imputation.
- **Validation**: After imputation, re-assess distributions, correlations, and model performance to ensure no artifacts were introduced.

## References
- [CRISP-DM Methodology Guide](documentation/CRISP-DM_Methodology_Guide.md): Emphasizes cleaning data to avoid "garbage-in, garbage-out."
- [Data Preparation Notebook](notebooks/Lalita/data-preparation.ipynb): Contains code for indicator column imputation.
- [NHS PROMs Data Understanding](documentation/application-project-nhs-proms-master/notebooks/1.0-data-understanding.ipynb): Discusses missing value assessment and initial dropping strategy.

## Future Considerations
- Monitor for changes in missingness patterns with new data.
- Consider advanced techniques like Multiple Imputation by Chained Equations (MICE) for complex missingness.
- Document any updates to these decisions in model versioning and audits.

**Last Updated**: Based on data preparation as of the latest notebook execution.  
**Team Lead**: Data Lead (All Team Members)