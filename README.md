# Automated Twitter Scraper and Report Generator

This project is a Python script that automatically scrapes tweets, analyzes them, and generates a daily summary report sent via email. It leverages several powerful tools, including Apify for data collection and the Gemini API for intelligent topic analysis.

The entire process is automated using a GitHub Actions workflow, making it a "set it and forget it" solution for monitoring specific topics or keywords on Twitter.

### Features

* **Daily Data Collection:** Automatically scrapes tweets for defined search terms using an Apify actor.
* **Intelligent Topic Analysis:** Uses the Gemini API to analyze the collected tweets and identify the main topics and themes.
* **Comprehensive Email Report:** Generates an HTML report that includes key metrics, top tweets, and an hourly distribution chart.
* **Data Persistence:** Saves the raw tweet data as a CSV file in the repository's `output/` directory, with an option to upload it as a GitHub Actions artifact.
* **Customizable:** Easily configure search terms, API keys, and email settings using environment variables and GitHub Secrets.

---

### How It Works

1.  **GitHub Actions:** A scheduled GitHub Actions workflow (defined in `cron.yaml`) triggers the `script.py` once a day.
2.  **Apify Scraper:** The Python script connects to the Apify platform using your `APIFY_TOKEN` and runs a Twitter scraper actor. It collects tweets from the past 24 hours based on the `SEARCH_TERMS` you provide.
3.  **Data Processing:** The script processes the collected data using the `pandas` library to calculate metrics like total tweets, impressions, and interactions. It identifies the most-viewed tweets and top accounts.
4.  **AI Analysis:** If a `GEMINI_API_KEY` is configured, the script uses the Gemini API to perform a semantic analysis of the tweets and extract the most prominent topics.
5.  **Report Generation:** A daily HTML report is crafted, including a bar chart of hourly tweet activity, the top topics from the AI analysis, and tables of key tweets.
6.  **Email Delivery:** The report is sent to the designated recipient(s) using SMTP, embedding the chart as an image directly in the email body.
7.  **Data Archiving:** The raw tweet data is saved as a CSV file and committed back to the repository by the GitHub Action.

---

### Setup

To use this project, you need to set up several environment variables as GitHub Secrets in your repository.

#### 1. GitHub Secrets

Go to your repository's **Settings > Secrets and variables > Actions** and add the following secrets:

* **`APIFY_TOKEN`**: Your API token from Apify.
* **`ACTOR_ID`** (or `APIFY_ACTOR_ID`): The ID of the Apify Twitter scraper actor you want to use. The default is `apidojo/twitter-scraper-lite`.
* **`GEMINI_API_KEY`**: Your API key for the Google Gemini API.
* **`EMAIL_HOST`**: The SMTP host for sending emails (e.g., `smtp.gmail.com`).
* **`EMAIL_PORT`**: The SMTP port (e.g., `587`).
* **`EMAIL_USER`**: The email address you will use to send the report.
* **`EMAIL_PASSWORD`**: The password or application-specific password for the sender email.
* **`EMAIL_RECIPIENT`**: The email address(es) to receive the report. You can use a comma-separated list for multiple recipients.

#### 2. GitHub Variables

Add the following variable in the same section, under **Variables**:

* **`SEARCH_TERMS`**: A comma-separated list of keywords or phrases to search for on Twitter (e.g., `mercado libre, mercadolibre`).

#### 3. Workflow Trigger

The workflow is configured to run daily at 23:00 UTC, but you can also trigger it manually from the **Actions** tab by selecting the "Run Twitter scraper daily..." workflow and clicking **Run workflow**.

---

### Customization

* **Cron Schedule:** Modify the `cron` expression in `.github/workflows/cron.yaml` to change the daily run time.
* **Search Terms:** Change the `SEARCH_TERMS` variable directly in your repository's settings.
* **Email Content:** The HTML structure and content of the email can be modified in the `html_body` variable within `script.py`.
* **AI Analysis:** The `extraer_temas_generales_con_ia` function can be adjusted to change the prompt or the number of topics to extract.
