# CRISP-DM Methodology Guide
## Cross-Industry Standard Process for Data Mining

---

## 🎯 Business Understanding

Any good project starts with a deep understanding of the customer's needs. The Business Understanding phase focuses on understanding the objectives and requirements of the project:

### Determine business objectives
From a business perspective, what does CFGS really want to accomplish and then define business success criteria. For example, why is it important for CFGS to make their forecast less subjective?

### Assess situation
Determine resources availability, project requirements, assess risks and contingencies, and conduct a cost-benefit analysis. How can the business context of the case be used in the analysis of the data?

### Determine data mining goals
In addition to defining the business objectives, you should also define what success looks like from a technical data mining perspective. What insights can be gained from the sales data? What can we learn from item returns; can these be predicted?

### Produce project plan
Select technologies and tools and define detailed plans for each project phase.

While many teams hurry through this phase, establishing a strong business understanding is like building the foundation of a house – absolutely essential.

---

## 📊 Data Understanding

All data is collected in GitHub, you will find more information on the bottom of this page. Consider how this data can help you in solving this challenge:

### Collect initial data
Acquire the necessary data and (if necessary) load it into your analysis tool. How complete is the data?

### Describe data
Examine the data and document its surface properties like data format, number of records, or field identities.

### Verify data quality
How clean/dirty is the data? Document any quality issues.

### Explore data
Dig deeper into the data. Query it, visualize it, and identify relationships among the data. What kind of business dynamics do you recognise in the data? Which patterns do you see in the historical sales (trend, seasonality, events, week patterns, etc.)? What are your hypothesis on the underlying causes for the patterns that you find?

---

## 🔧 Data Preparation

This phase, which is often referred to as "data munging", prepares the final data set(s) for modelling. It has five tasks:

### Select data
Determine which data sets will be used and document reasons for inclusion/exclusion.

### Clean data
Often this is the lengthiest task. Without it, you'll likely fall victim to garbage-in, garbage-out. A common practice during this task is to correct, impute, or remove erroneous values.

### Construct data
Derive new attributes that will be helpful. For example, convert the day, month, and year features to a single date feature.

### Integrate data
Create new data sets by combining data from multiple sources.

### Format data
Re-format data as necessary. For example, you might convert string values that store numbers to numeric values so that you can perform mathematical operations. Additionally, downcasting data can reduce the number of megabytes used by the data.

---

## 🤖 Modelling

Here you'll likely build and assess various models based on several different modeling techniques. This phase has four tasks:

### Select modelling techniques
Determine which algorithms to try (e.g. regression, neural net).

### Generate test design
Pending your modeling approach, you might need to split the data into training, test, and validation sets.

### Build model
As glamorous as this might sound, this might just be executing a few lines of code like:

```python
reg = LinearRegression().fit(X, y)
```

### Assess model
Generally, multiple models are competing against each other, and the data scientist needs to interpret the model results based on domain knowledge, the pre-defined success criteria, and the test design.

Although the CRISP-DM guide suggests to "iterate model building and assessment until you strongly believe that you have found the best model(s)", in practice you should continue iterating until you find a "good enough" model, proceed through the CRISP-DM lifecycle, then further improve the model in future iterations.

---

## 📈 Evaluation

Whereas the Assess Model task of the Modelling phase focuses on technical model assessment, the Evaluation phase looks more broadly at which model best meets the business and what to do next. This phase has three tasks:

### Evaluate results
Do the models meet the business success criteria? Which one(s) should we approve for the business?

### Review process
Review the work accomplished. Was anything overlooked? Were all steps properly executed? Summarise findings and correct anything if needed.

### Determine next steps
Based on the previous three tasks, determine whether to proceed to deployment, iterate further, or initiate new projects.

---

## 🚀 Deployment

Depending on the requirements, the deployment phase can be as simple as generating a report or as complex as implementing a repeatable data mining process across the enterprise. A model is not particularly useful unless the customer can access its results. The complexity of this phase varies widely. This final phase has four tasks:

### Plan deployment
Develop and document a plan for deploying the model.

### Plan monitoring and maintenance
Develop a thorough monitoring and maintenance plan to avoid issues during the operational phase (or post-project phase) of a model.

### Produce final report
The project team documents a summary of the project which might include a final presentation of data mining results.

### Review project
Conduct a project retrospective about what went well, what could have been better, and how to improve in the future.

---

**Reference Document**  
**Created**: December 7, 2025  
**Team**: Project NHS - EAISI  
**Purpose**: Methodology reference for CRISP-DM implementation