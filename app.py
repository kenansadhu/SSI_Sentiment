import pandas as pd
from textblob import TextBlob
from flask import Flask, render_template, request, redirect, url_for, send_file
import io

app = Flask(__name__)
app.config['SECRET_KEY'] = 'some_secret_key'

# Global variables
df_global = None
stats_global = None
sentiment_counts_global = None

# For dynamic dropdowns
data_dict_global = {}

# We will also store overall sentiment examples in global variables
pos_examples_global = None
neg_examples_global = None
neu_examples_global = None

NUMERIC_COLS = [
    "Online_Platforms",
    "M_Flex_Overview",
    "Learning_Experience",
    "No flexible learning activities",
    "Study independently the material uploaded on the UPH learning platform",
    "Work on assignments/practice questions independently",
    "Discuss in online discussion forums",
    "Collaborate on tasks/projects collaboratively in groups",
    "Participate in tutorial activities with lecturers and tutors"
]

TEXT_COL = "Tuliskan saran/komentar anda pada kolom dibawah ini terkait pengalaman belajar Anda di UPH."
LOCATION_COL = "Lokasi Anda berkuliah :"
FACULTY_COL = "Faculty"
PROGRAM_COL = "Study_Program"

def get_polarity(text):
    """
    Return a float polarity score from -1.0 (very negative) to 1.0 (very positive).
    """
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

def analyze_sentiment(text):
    """
    Return a simple sentiment label (Positive, Negative, or Neutral)
    based on polarity.
    """
    pol = get_polarity(text)
    if pol > 0:
        return "Positive"
    elif pol < 0:
        return "Negative"
    else:
        return "Neutral"

def convert_sets_to_lists(d):
    """
    Recursively convert all set values in a nested dict to sorted lists
    so they can be JSON-serialized by Jinja's tojson filter.
    """
    new_dict = {}
    for key, val in d.items():
        if isinstance(val, dict):
            new_dict[key] = convert_sets_to_lists(val)
        elif isinstance(val, set):
            new_dict[key] = sorted(val)
        else:
            new_dict[key] = val
    return new_dict

def get_sentiment_examples(df, text_col=TEXT_COL, sentiment_col="Sentiment", polarity_col="Polarity"):
    """
    Returns up to 5 examples each for Positive, Negative, and Neutral.
    Positive = highest polarity
    Negative = lowest polarity
    Neutral = first 5 in ascending order of polarity (or any approach).
    """
    if df.empty:
        return None, None, None

    # Positive: sort descending by polarity, take top 5
    pos_df = df[df[sentiment_col] == "Positive"].sort_values(by=polarity_col, ascending=False).head(5)
    # Negative: sort ascending by polarity, take top 5
    neg_df = df[df[sentiment_col] == "Negative"].sort_values(by=polarity_col, ascending=True).head(5)
    # Neutral: just take top 5 (polarity ~ 0)
    neu_df = df[df[sentiment_col] == "Neutral"].head(5)

    return pos_df, neg_df, neu_df

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route: 
    - GET: Show the upload page (if no data loaded yet) 
    - POST: Upload the Excel file, do overall analysis, build data_dict for dynamic dropdowns
    """
    global df_global, stats_global, sentiment_counts_global
    global data_dict_global
    global pos_examples_global, neg_examples_global, neu_examples_global

    if request.method == 'POST':
        if 'excel_file' not in request.files:
            return "No file part in the request."
        file = request.files['excel_file']
        if file.filename == '':
            return "No selected file."

        # Read Excel into a DataFrame
        try:
            df = pd.read_excel(file)
        except Exception as e:
            return f"Error reading Excel file: {e}"

        # Compute numeric stats
        numeric_cols = [col for col in NUMERIC_COLS if col in df.columns]
        stats = df[numeric_cols].describe() if numeric_cols else None

        # Compute polarity & sentiment
        if TEXT_COL in df.columns:
            df["Polarity"] = df[TEXT_COL].apply(get_polarity)
            df["Sentiment"] = df[TEXT_COL].apply(analyze_sentiment)
            sentiment_counts = df["Sentiment"].value_counts(dropna=False)
        else:
            df["Polarity"] = 0.0
            df["Sentiment"] = "N/A"
            sentiment_counts = None

        # Build the dictionary for dynamic dropdowns (location->faculty->program)
        data_dict = {}
        if all(col in df.columns for col in [LOCATION_COL, FACULTY_COL, PROGRAM_COL]):
            for _, row in df.iterrows():
                loc = row[LOCATION_COL]
                fac = row[FACULTY_COL]
                sp  = row[PROGRAM_COL]
                if pd.notna(loc) and pd.notna(fac) and pd.notna(sp):
                    data_dict.setdefault(loc, {}).setdefault(fac, set()).add(sp)

        data_dict = convert_sets_to_lists(data_dict)

        # Get top 5 examples for each sentiment in the overall data
        pos_df, neg_df, neu_df = get_sentiment_examples(df)
        # Convert them to HTML if not None
        if pos_df is not None:
            pos_html = pos_df[[TEXT_COL, "Polarity"]].to_html(
                classes="table table-sm table-striped", 
                index=False, 
                header=["Comment", "Polarity"]
            )
            neg_html = neg_df[[TEXT_COL, "Polarity"]].to_html(
                classes="table table-sm table-striped", 
                index=False,
                header=["Comment", "Polarity"]
            )
            neu_html = neu_df[[TEXT_COL, "Polarity"]].to_html(
                classes="table table-sm table-striped", 
                index=False,
                header=["Comment", "Polarity"]
            )
        else:
            pos_html = neg_html = neu_html = None

        # Store globally
        df_global = df
        stats_global = stats
        sentiment_counts_global = sentiment_counts
        data_dict_global = data_dict
        pos_examples_global = pos_html
        neg_examples_global = neg_html
        neu_examples_global = neu_html

        return render_template(
            'index.html',
            df_exists=True,
            overall_stats=stats.to_html(classes="table table-striped", justify="left") if stats is not None else None,
            overall_sentiment_counts=(sentiment_counts.to_dict() if sentiment_counts is not None else None),
            # Overall examples
            pos_examples=pos_html,
            neg_examples=neg_html,
            neu_examples=neu_html,
            # Filtered
            filtered_stats=None,
            filtered_sentiment_counts=None,
            filtered_pos_examples=None,
            filtered_neg_examples=None,
            filtered_neu_examples=None,
            filtered_df=None,
            filter_applied=False,
            data_dict=data_dict_global
        )

    # If GET
    if df_global is None:
        # No data loaded
        return render_template('index.html', df_exists=False)

    # If we already have data, show the "overall" page
    return render_template(
        'index.html',
        df_exists=True,
        overall_stats=(
            stats_global.to_html(classes="table table-striped", justify="left") 
            if stats_global is not None else None
        ),
        overall_sentiment_counts=(
            sentiment_counts_global.to_dict() 
            if sentiment_counts_global is not None else None
        ),
        pos_examples=pos_examples_global,
        neg_examples=neg_examples_global,
        neu_examples=neu_examples_global,
        # Filter placeholders
        filtered_stats=None,
        filtered_sentiment_counts=None,
        filtered_pos_examples=None,
        filtered_neg_examples=None,
        filtered_neu_examples=None,
        filtered_df=None,
        filter_applied=False,
        data_dict=data_dict_global
    )

@app.route('/filter', methods=['POST'])
def filter_data():
    """
    Apply filters based on user selection and display the filtered result.
    Also show separate stats for the filtered subset.
    """
    global df_global, stats_global, sentiment_counts_global, data_dict_global
    if df_global is None:
        return redirect(url_for('index'))

    location_input = request.form.get("location_input", "").strip()
    faculty_input  = request.form.get("faculty_input", "").strip()
    study_program_input = request.form.get("study_program_input", "").strip()

    filtered_df = df_global.copy()
    if location_input:
        filtered_df = filtered_df[filtered_df[LOCATION_COL] == location_input]
    if faculty_input:
        filtered_df = filtered_df[filtered_df[FACULTY_COL] == faculty_input]
    if study_program_input:
        filtered_df = filtered_df[filtered_df[PROGRAM_COL] == study_program_input]

    # Filtered numeric stats
    numeric_cols = [col for col in NUMERIC_COLS if col in filtered_df.columns]
    if not filtered_df.empty and numeric_cols:
        filtered_stats = filtered_df[numeric_cols].describe()
    else:
        filtered_stats = None

    # Filtered sentiment counts
    if not filtered_df.empty and "Sentiment" in filtered_df.columns:
        filtered_sentiment_counts = filtered_df["Sentiment"].value_counts(dropna=False)
    else:
        filtered_sentiment_counts = None

    # Filtered top 5 sentiment examples
    pos_df, neg_df, neu_df = get_sentiment_examples(filtered_df)
    if pos_df is not None:
        pos_html = pos_df[[TEXT_COL, "Polarity"]].to_html(
            classes="table table-sm table-striped", 
            index=False, 
            header=["Comment", "Polarity"]
        )
        neg_html = neg_df[[TEXT_COL, "Polarity"]].to_html(
            classes="table table-sm table-striped", 
            index=False,
            header=["Comment", "Polarity"]
        )
        neu_html = neu_df[[TEXT_COL, "Polarity"]].to_html(
            classes="table table-sm table-striped", 
            index=False,
            header=["Comment", "Polarity"]
        )
    else:
        pos_html = neg_html = neu_html = None

    # Show the first 50 rows in the HTML table
    table_html = None
    if not filtered_df.empty:
        table_html = filtered_df.head(50).to_html(
            classes="table table-bordered table-hover", 
            index=False
        )

    # Re-render template with both overall + filtered data
    return render_template(
        'index.html',
        df_exists=True,
        # Overall (unchanged)
        overall_stats=(
            stats_global.to_html(classes="table table-striped", justify="left") 
            if stats_global is not None else None
        ),
        overall_sentiment_counts=(
            sentiment_counts_global.to_dict() 
            if sentiment_counts_global is not None else None
        ),
        # We'll just hide them in the HTML via toggle
        pos_examples=None,
        neg_examples=None,
        neu_examples=None,
        # Filtered
        filtered_stats=(
            filtered_stats.to_html(classes="table table-striped", justify="left") 
            if filtered_stats is not None else None
        ),
        filtered_sentiment_counts=(
            filtered_sentiment_counts.to_dict() 
            if filtered_sentiment_counts is not None else None
        ),
        filtered_pos_examples=pos_html,
        filtered_neg_examples=neg_html,
        filtered_neu_examples=neu_html,
        filtered_df=table_html,
        filter_applied=True,
        filter_count=len(filtered_df),
        data_dict=data_dict_global
    )

@app.route('/export', methods=['POST'])
def export_data():
    """
    Export the filtered data as an Excel file download.
    We apply the same filters used in /filter.
    """
    global df_global
    if df_global is None:
        return redirect(url_for('index'))

    location_input = request.form.get("location_input", "").strip()
    faculty_input  = request.form.get("faculty_input", "").strip()
    study_program_input = request.form.get("study_program_input", "").strip()

    filtered_df = df_global.copy()
    if location_input:
        filtered_df = filtered_df[filtered_df[LOCATION_COL] == location_input]
    if faculty_input:
        filtered_df = filtered_df[filtered_df[FACULTY_COL] == faculty_input]
    if study_program_input:
        filtered_df = filtered_df[filtered_df[PROGRAM_COL] == study_program_input]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        filtered_df.to_excel(writer, index=False, sheet_name="Filtered")
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="filtered_data.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

if __name__ == '__main__':
    app.run(debug=True)
