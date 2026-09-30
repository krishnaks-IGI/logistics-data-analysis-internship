# Pune Last-Mile Logistics Analytics

A four-week learning project that follows a logistics data analyst's workflow: plan the questions, prepare shipment data, explain patterns with charts, and explore delivery-time prediction.

> **Data note:** All shipment rows and results in this project are synthetic examples created for practice. They are not YuvaIntern, company, or customer data, and should not be read as real operational performance.

## Weekly reports

| Week | Focus | Report |
| --- | --- | --- |
| 1 | Strategic planning for a last-mile logistics analysis | [Week 1 report](Week_1_Strategic_Planning_Logistics.docx) |
| 2 | Data collection, quality checks, cleaning, and preparation | [Week 2 report](Week_2_Data_Collection_Cleaning_Preprocessing.docx) |
| 3 | Exploratory analysis and visual explanations | [Week 3 report](Week_3_Advanced_Analysis_Visualization_Logistics.docx) |
| 4 | Delivery-time prediction and practical optimization ideas | [Week 4 report](Week_4_Predictive_Modeling_Optimization_Logistics.docx) |

## Project files

- `logistics_analysis_pipeline.py` creates a reproducible synthetic sample, applies the documented preparation steps, summarizes the data, compares a simple baseline with a regression model, and saves charts.
- `synthetic_pune_last_mile_shipments_raw.csv` is the generated raw example, including deliberate quality issues for practice.
- `synthetic_pune_last_mile_shipments.csv` is the cleaned example used in the reports.
- `requirements.txt` lists the Python packages used by the script.

## Run it

You will need Python 3.10 or newer. From this folder, install the packages and run the pipeline:

```bash
python -m pip install -r requirements.txt
python logistics_analysis_pipeline.py
```

The script uses a fixed random seed, so it can recreate the example data and charts consistently. Running it replaces the two generated CSV files and writes chart images to a `charts/` folder.

## How to read the results

The analysis uses shipment and delivery-time examples to show how data quality affects conclusions. It compares a training-set median estimate with a linear regression model on later dates, then discusses where patterns could guide further investigation.

Because the data is simulated, the findings are for learning and demonstration only. They do not establish that a particular operational change will improve real deliveries. Any real deployment would need verified company data, stakeholder review, and validation against current operations.
