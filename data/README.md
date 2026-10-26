# SMS Spam Collection data

Source: Almeida, T. and Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

Dataset page: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

Licence listed by UCI: Creative Commons Attribution 4.0 International (https://creativecommons.org/licenses/by/4.0/).

The `raw/SMSSpamCollection` file is the unchanged official download retrieved on 26 October 2026. `raw/UCI_README.txt` preserves the accompanying source documentation. Original SMS messages can contain offensive language.

SHA-256 of `SMSSpamCollection`:
`7d039a24a6083ed9ef0f806ebad56bbb976e3aeb8de05669173bfdc4996c239d`

Each physical line is one record. Split at the first tab only; quotation marks inside messages are literal text, not CSV delimiters. The file contains 5,574 records. `source_line` is a one-based source-file line number for traceability.


## Processed export

`processed/sms_cleaned.csv` contains 5,171 unique messages after 403 exact duplicate messages are removed. There are no missing or blank records in this source. The exported columns are `source_line`, `label`, `message` and `target` (ham = 0, spam = 1). The first occurrence of each exact message is retained without changing its text. These cleaning transformations are the only changes to the source records. See `reports/dataset_summary.json` for counts.

Regenerate the processed CSV and summary by running `sms_spam_detection.ipynb` from the repository root. The original dataset attribution and licence also apply to the processed data.
