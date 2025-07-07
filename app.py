import os
import uuid
from datetime import datetime
from io import BytesIO
import glob
import base64

import pandas as pd
import numpy as np
from flask import (
    Flask, request, render_template, redirect, url_for,
    send_file, flash, session, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.figure_factory as ff

from bs4 import BeautifulSoup
import re
import emoji

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import contractions

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize app and folders
app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static/plots'
USER_DB = 'users.csv'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Global variables (simulate session storage for demo)
users = {}
df = None
original_df = None
cleaning_report = []
cleaning_code_lines = []
data_versions = []
cleaned_columns = set()
column_previews = {}
smart_chart_results = []
column_versions = {}  




selected_group_col = None
selected_features = []
selected_chart_type = "histogram"
selected_chart_template = "plotly_dark"

# ----------------------
# Helper Functions
# ----------------------
def export_pipeline_code():
    return "\n".join(cleaning_code_lines)

def log_action(message):
    if 'username' in session:
        user_log = f"logs_{session['username']}.txt"
        with open(user_log, "a") as f:
            f.write(f"[{datetime.now()}] {message}\n")

def add_version(data):
    version = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": data.copy()
    }
    data_versions.append(version)
def load_csv_in_chunks(filepath, chunk_size=10000):
    chunks = []
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    return df

# Text cleaning helpers
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text_basic(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    text = contractions.fix(text)
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = ''.join(c for c in text if c not in emoji.UNICODE_EMOJI['en'])
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

def remove_outliers(df, col):
    if col not in df.columns:
        return df
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    filter = (df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)
    return df.loc[filter]

def fill_na_with_mean(df, col):
    if col in df.columns:
        mean_val = df[col].mean()
        df[col] = df[col].fillna(mean_val)
    return df

def fill_na_with_median(df, col):
    if col in df.columns:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    return df

def fill_na_with_mode(df, col):
    if col in df.columns:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
    return df

def drop_na(df, col):
    if col in df.columns:
        df = df.dropna(subset=[col])
    return df

def convert_dtype(df, col):
    if col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass
    return df

stemmer = PorterStemmer()

def apply_text_cleaning_steps(series, steps):
    for step in steps:
        if step == 'lowercase':
            series = series.str.lower()
        elif step == 'remove_punct':
            series = series.str.replace(r'[^\w\s]', '', regex=True)
        elif step == 'remove_stopwords':
            series = series.apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]) if isinstance(x, str) else x)
        elif step == 'lemmatize':
            series = series.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]) if isinstance(x, str) else x)
        elif step == 'stem':
            series = series.apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]) if isinstance(x, str) else x)
        elif step == 'remove_html':
            series = series.apply(lambda x: BeautifulSoup(x, "html.parser").get_text() if isinstance(x, str) else x)
        elif step == 'remove_emoji':
            series = series.apply(lambda x: ''.join(c for c in x if c not in emoji.UNICODE_EMOJI['en']) if isinstance(x, str) else x)
        elif step == 'remove_digits':
            series = series.str.replace(r'\d+', '', regex=True)
        elif step == 'deduplicate_chars':
            series = series.apply(lambda x: re.sub(r'(.)\1+', r'\1', x) if isinstance(x, str) else x)
        elif step == 'strip_whitespace':
            series = series.str.strip()
            series = series.str.replace(r'\s+', ' ', regex=True)
    return series


def cleanup_old_plots(folder=STATIC_FOLDER, pattern="pairplot_*.png"):
    files = glob.glob(os.path.join(folder, pattern))
    for f in files:
        try:
            os.remove(f)
        except Exception:
            pass

def generate_plotly_chart(df, chart_type, group_col, features, template):
    import plotly.express as px
    import plotly.graph_objects as go
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    import seaborn as sns
    import pandas as pd
    from collections import Counter
    import re

    color = group_col if group_col in df.columns and group_col else None

    if chart_type == "histogram":
        fig = px.histogram(df, x=features[0], color=color, template=template)

    elif chart_type == "box":
        fig = px.box(df, x=color, y=features[0], template=template) if color else px.box(df, y=features[0], template=template)

    elif chart_type == "violin":
        fig = px.violin(df, x=color, y=features[0], box=True, template=template) if color else px.violin(df, y=features[0], box=True, template=template)

    elif chart_type == "scatter":
        if len(features) >= 2:
            fig = px.scatter(df, x=features[0], y=features[1], color=color, template=template)
        else:
            raise ValueError("Please select at least two numeric features for scatter plot.")

    elif chart_type == "density":
        fig = px.density_contour(df, x=features[0], y=features[1] if len(features) > 1 else None, template=template)

    elif chart_type == "line":
        fig = px.line(df, x=features[0], y=features[1] if len(features) > 1 else features[0], color=color, template=template)

    elif chart_type == "bar":
        fig = px.bar(df, x=features[0], color=color, template=template)

    elif chart_type == "count":
        fig = px.histogram(df, x=features[0], color=color, barmode='group', template=template)

    elif chart_type == "pie":
        fig = px.pie(df, names=features[0], template=template)

    elif chart_type == "stackedbar":
        fig = px.histogram(df, x=features[0], color=color, barmode='stack', template=template)

    elif chart_type == "heatmap":
        corr = df.select_dtypes(include='number').corr()
        fig = px.imshow(corr, text_auto=True, aspect='auto', color_continuous_scale='RdBu_r', template=template)

    elif chart_type == "pairplot":
        sns.set_theme(style="darkgrid")
        pairplot = sns.pairplot(df[features].dropna())
        buf = BytesIO()
        pairplot.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        encoded = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{encoded}" style="max-width:100%;"/>'

    elif chart_type == "scatter_matrix":
        fig = px.scatter_matrix(df, dimensions=features[:5], color=color, template=template)

    elif chart_type == "wordcloud":
        text = " ".join(df[features[0]].dropna().astype(str))
        wc = WordCloud(width=800, height=400, background_color='black', colormap='Pastel1').generate(text)
        buf = BytesIO()
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode()
        return f'<img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/>'

    elif chart_type == "topwords":
        words = " ".join(df[features[0]].dropna().astype(str))
        words = re.findall(r'\b\w+\b', words.lower())
        top = Counter(words).most_common(15)
        labels, values = zip(*top)
        fig = px.bar(x=labels, y=values, labels={'x': 'Word', 'y': 'Frequency'}, template=template)

    else:
        fig = go.Figure()
        fig.add_annotation(text="Invalid chart type selected", showarrow=False)

    fig.update_layout(margin=dict(t=30, b=20, l=10, r=10))
    return fig.to_html(full_html=False)


from functools import wraps
from flask import session, redirect, url_for, flash

# ----------------------
# Login Required Decorator
# ----------------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash("Please log in to access this page.", "warning")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ----------------------
# Routes
# ----------------------

@app.route('/')
def home():
    login_success = session.pop('login_success', False)
    return render_template('home.html', login_success=login_success)




@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload_file():
    global df, original_df, cleaning_report, cleaning_code_lines, data_versions
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            df = load_csv_in_chunks(file_path)
            original_df = df.copy()
            cleaning_report.clear()
            cleaning_code_lines.clear()
            data_versions.clear()
            cleaned_columns.clear()
            add_version(df)
            flash('File uploaded successfully')
            return redirect(url_for('preview'))
        else:
            flash('Please upload a CSV file')
            return redirect(request.url)
    return render_template("upload.html", step=1)

@app.route('/preview', methods=['GET', 'POST'])
@login_required
def preview():
    global df, cleaning_report, selected_chart_type, selected_group_col, selected_features, selected_chart_template, smart_chart_results
    invalid_rows = []
    validation_rules = session.get('validation_rules', [])




    if df is None:
        flash('Please upload a dataset first')
        return redirect(url_for('upload_file'))

    step = 2

    # Initialize or retrieve chart session
    if 'chart_history' not in session:
        session['chart_history'] = []

    chart_html = None

    if request.method == 'POST':
        selected_chart_type = request.form.get('chart_type', 'histogram')
        selected_group_col = request.form.get('group_col', None)
        selected_chart_template = request.form.get('chart_template', 'plotly_dark')
        selected_features = request.form.getlist('features')

        try:
            chart_html = generate_plotly_chart(df, selected_chart_type, selected_group_col, selected_features, selected_chart_template)
            session['chart_history'].append(chart_html)
            session.modified = True
        except Exception as e:
            flash(f"Chart error: {str(e)}")
    else:
        selected_chart_type = 'histogram'
        selected_group_col = None
        selected_chart_template = 'plotly_dark'
        selected_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Column type detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=[object]).columns.tolist()
    groupable_cols = df.columns.tolist()

    # Smart chart recommendations (if no manual charts exist)
    recommended_charts = []
    if request.method == 'GET' and not session['chart_history']:
        chart_templates = ['plotly_dark']

        for col in numeric_cols:
            recommended_charts.extend([
                {
                    "html": generate_plotly_chart(df, "histogram", None, [col], chart_templates[0]),
                    "title": f"📶 Histogram of '{col}'",
                    "reason": "Histograms show the distribution and frequency of values in a numerical column."
                },
                {
                    "html": generate_plotly_chart(df, "box", None, [col], chart_templates[0]),
                    "title": f"📦 Box Plot of '{col}'",
                    "reason": "Box plots help detect outliers and understand spread (median, quartiles)."
                },
                {
                    "html": generate_plotly_chart(df, "violin", None, [col], chart_templates[0]),
                    "title": f"🎻 Violin Plot of '{col}'",
                    "reason": "Violin plots combine box plots with KDE distributions."
                }
            ])

        if len(numeric_cols) >= 2:
            recommended_charts.append({
                "html": generate_plotly_chart(df, "scatter", None, numeric_cols[:2], chart_templates[0]),
                "title": f"🟢 Scatter Plot: '{numeric_cols[0]}' vs '{numeric_cols[1]}'",
                "reason": "Scatter plots help visualize relationships or correlations between two numeric variables."
            })

        for col in text_cols:
            if df[col].nunique() < 50:
                recommended_charts.extend([
                    {
                        "html": generate_plotly_chart(df, "count", None, [col], chart_templates[0]),
                        "title": f"📊 Count Plot of '{col}'",
                        "reason": "Shows frequency of each category."
                    },
                    {
                        "html": generate_plotly_chart(df, "bar", None, [col], chart_templates[0]),
                        "title": f"📋 Bar Chart of '{col}'",
                        "reason": "Bar charts compare category sizes clearly."
                    }
                ])
            recommended_charts.extend([
                {
                    "html": generate_plotly_chart(df, "wordcloud", None, [col], chart_templates[0]),
                    "title": f"🔤 Word Cloud for '{col}'",
                    "reason": "Highlights frequent words in a column visually."
                },
                {
                    "html": generate_plotly_chart(df, "topwords", None, [col], chart_templates[0]),
                    "title": f"🔢 Top Words in '{col}'",
                    "reason": "Shows most frequent words as a bar chart."
                }
            ])

        if len(numeric_cols) >= 2:
            recommended_charts.extend([
                {
                    "html": generate_plotly_chart(df, "heatmap", None, numeric_cols, chart_templates[0]),
                    "title": "🧪 Correlation Heatmap",
                    "reason": "Visualize relationships between all numerical features."
                },
                {
                    "html": generate_plotly_chart(df, "pairplot", None, numeric_cols[:5], chart_templates[0]),
                    "title": "🧬 Pair Plot of Numerical Features",
                    "reason": "Explore scatter relationships between all feature pairs."
                }
            ])

    # Summary Statistics
    desc_stats = df.describe(include='all').to_html(classes='table table-striped table-bordered')

    def get_col_type(series):
        if pd.api.types.is_numeric_dtype(series):
            return "Numerical"
        elif pd.api.types.is_string_dtype(series):
            return "Categorical"
        return "Mixed"

    def has_outliers(series):
        if pd.api.types.is_numeric_dtype(series):
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            return ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).any()
        return False
    

    # Cleaning Suggestions
    cleaning_suggestions = []
    for col in df.columns:
        col_type = get_col_type(df[col])
        issues = []
        if col not in cleaned_columns:
            if df[col].isnull().sum() > 0:
                issues.append("Contains missing values")
            if col_type == "Numerical" and has_outliers(df[col]):
                issues.append("Contains outliers")

        cleaning_suggestions.append({
            'column': col,
            'type': col_type,
            'suggestions': issues
        })

    # Step control
    if cleaning_report:
        session['cleaned'] = True
    if session.get('cleaned'):
        step = max(step, 3)
    if session.get('downloaded'):
        step = max(step, 4)

    
        # 🧪 Data Validation Logic
    validation_rules = session.get('validation_rules', [])
    violating_rows = session.get('violating_rows')

    # Evaluate rules only if violating rows not already present
    if validation_rules and (violating_rows is None):

        try:
            mask = pd.Series([True] * len(df), index=df.index)

            for rule in validation_rules:
                col = rule['column']
                op = rule['condition']
                val = rule['value']

                if op in ['>', '<', '==', '!=']:
                    val = float(val)
                    if op == '>':
                        mask &= ~(df[col] > val)
                    elif op == '<':
                        mask &= ~(df[col] < val)
                    elif op == '==':
                        mask &= ~(df[col] == val)
                    elif op == '!=':
                        mask &= ~(df[col] != val)

                elif op == 'between':
                    low, high = map(float, val.split(','))
                    mask &= ~df[col].between(low, high)

                elif op == 'contains':
                    mask &= ~df[col].astype(str).str.contains(str(val), na=False)

                elif op == 'not_contains':
                    mask &= df[col].astype(str).str.contains(str(val), na=False)

            violating_rows = df[~mask].copy()

        except Exception:
            violating_rows = None



    # Smart charts from stored results
    smart_charts = smart_chart_results

    # Dataset Summary Panel
    summary_panel = {
        'total_rows': len(df),
        'total_cols': df.shape[1],
        'num_numerical': len(df.select_dtypes(include=['number']).columns),
        'num_categorical': len(df.select_dtypes(include=['object', 'category']).columns),
        'num_text': len(df.select_dtypes(include=['object']).columns),
        'missing_pct': round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
        'duplicates': df.duplicated().sum(),
        'memory_usage': round(df.memory_usage(deep=True).sum() / (1024**2), 2)
    }

    return render_template(
        'preview.html',
        step=step,
        tables=[df.head(10).to_html(classes='table table-striped table-bordered')],
        profile={'desc_stats': desc_stats},
        cleaning_suggestions=cleaning_suggestions,
        cleaned_columns=cleaned_columns,
        cleaning_report=cleaning_report,
        interactive_charts=session['chart_history'],
        numeric_columns=numeric_cols,
        text_columns=text_cols,
        groupable_columns=groupable_cols,
        selected_chart_type=selected_chart_type,
        selected_group_col=selected_group_col,
        selected_features=selected_features,
        selected_chart_template=selected_chart_template,
        cleaning_code="\n".join(cleaning_code_lines),
        recommended_charts=recommended_charts,
        smart_charts=smart_charts,
        summary_panel=summary_panel,
        validation_rules=validation_rules,
        violating_rows=violating_rows,df=df)

@app.route('/smart_visualizer', methods=['POST'])
@login_required
def smart_visualizer():
    global df
    if df is None:
        flash("Please upload a dataset first.")
        return redirect(url_for('upload_file'))

    features = request.form.getlist('features')
    group_col = request.form.get('group_col')
    theme = request.form.get('smart_theme', 'plotly_dark')
    viz_focus = request.form.get('viz_focus', 'auto')
    chart_type = request.form.get('manual_chart_type')
    action = request.form.get('action')  # manual or auto

    smart_charts = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=[object]).columns.tolist()

    if len(features) == 0:
        flash("Please select at least one feature.")
        return redirect(url_for('preview'))

    if action == 'manual':
        if not chart_type:
            flash("Please select a chart type to add manually.")
            return redirect(url_for('preview'))

        smart_charts.append({
            "html": generate_plotly_chart(df, chart_type, group_col, features, theme),
            "title": f"Custom Chart: {chart_type.title()}",
            "reason": f"User manually selected this chart type for: {', '.join(features)}."
        })

    elif action == 'auto':
        for col in features:
            if col in numeric_cols:
                if viz_focus in ['auto', 'distribution']:
                    smart_charts.append({
                        "html": generate_plotly_chart(df, "histogram", group_col, [col], theme),
                        "title": f"📊 Histogram of '{col}'",
                        "reason": "Shows the distribution of a numeric column."
                    })
                    smart_charts.append({
                        "html": generate_plotly_chart(df, "box", group_col, [col], theme),
                        "title": f"📦 Box Plot of '{col}'",
                        "reason": "Helps identify outliers and value spread."
                    })
                    smart_charts.append({
                        "html": generate_plotly_chart(df, "violin", group_col, [col], theme),
                        "title": f"🎻 Violin Plot of '{col}'",
                        "reason": "Combines box plot with data distribution."
                    })

                if viz_focus in ['auto', 'correlation'] and len(features) >= 2 and all(f in numeric_cols for f in features):
                    smart_charts.append({
                        "html": generate_plotly_chart(df, "scatter", group_col, features[:2], theme),
                        "title": f"🟢 Scatter Plot: '{features[0]}' vs '{features[1]}'",
                        "reason": "Shows relationship between two numeric features."
                    })
                    smart_charts.append({
                        "html": generate_plotly_chart(df, "heatmap", None, features, theme),
                        "title": "🧪 Correlation Heatmap",
                        "reason": "Visualizes correlation between numeric features."
                    })
                    if len(features) > 2:
                        smart_charts.append({
                            "html": generate_plotly_chart(df, "pairplot", None, features[:5], theme),
                            "title": "🔁 Pair Plot of Selected Features",
                            "reason": "Shows all pairwise numeric relationships."
                        })

            elif col in text_cols:
                if viz_focus in ['auto', 'text']:
                    smart_charts.append({
                        "html": generate_plotly_chart(df, "wordcloud", None, [col], theme),
                        "title": f"🔤 Word Cloud for '{col}'",
                        "reason": "Highlights common words in text."
                    })
                    smart_charts.append({
                        "html": generate_plotly_chart(df, "topwords", None, [col], theme),
                        "title": f"🔢 Top Words in '{col}'",
                        "reason": "Shows most frequent words in a column."
                    })

                if viz_focus in ['auto', 'comparison'] and df[col].nunique() < 50:
                    smart_charts.append({
                        "html": generate_plotly_chart(df, "count", group_col, [col], theme),
                        "title": f"📊 Count Plot of '{col}'",
                        "reason": "Shows how often each category appears."
                    })
                    smart_charts.append({
                        "html": generate_plotly_chart(df, "bar", group_col, [col], theme),
                        "title": f"📋 Bar Chart of '{col}'",
                        "reason": "Compares category sizes."
                    })

    global smart_chart_results
    smart_chart_results = smart_charts

    return redirect(url_for('preview'))
@app.route('/add_validation_rule', methods=['POST'])
@login_required
def add_validation_rule():
    column = request.form.get('rule_column')
    operator = request.form.get('rule_condition')
    value = request.form.get('rule_value')
    description = request.form.get('description', '')

    if not column or not operator or value is None:
        flash("All fields (column, operator, value) are required.", "danger")
        return redirect(url_for('preview'))

    rule = {
        'column': column,
        'condition': operator,
        'value': value,
        'description': description
    }

    rules = session.get('validation_rules', [])
    rules.append(rule)
    session['validation_rules'] = rules
    session.modified = True

    flash(f"Rule added: {column} {operator} {value}", "success")
    return redirect(url_for('preview'))


@app.route('/apply_validation', methods=['POST'])
@login_required
def apply_validation():
    global df
    rules = session.get('validation_rules', [])
    violating_df = df.copy()

    for rule in rules:
        col = rule['column']
        cond = rule['condition']
        val = rule['value']

        try:
            if cond == '>':
                violating_df = violating_df[~(violating_df[col] > float(val))]
            elif cond == '<':
                violating_df = violating_df[~(violating_df[col] < float(val))]
            elif cond == '==':
                violating_df = violating_df[~(violating_df[col] == float(val))]
            elif cond == '!=':
                violating_df = violating_df[~(violating_df[col] != float(val))]
            elif cond == 'between':
                low, high = map(float, val.split(','))
                violating_df = violating_df[~violating_df[col].between(low, high)]
            elif cond == 'contains':
                violating_df = violating_df[~violating_df[col].astype(str).str.contains(val, na=False)]
            elif cond == 'not_contains':
                violating_df = violating_df[violating_df[col].astype(str).str.contains(val, na=False)]
        except Exception:
            continue  # Skip rule if error

    session['violating_rows'] = violating_df.to_dict(orient='records')
    flash(f"{len(violating_df)} rows violate the rules.", "warning")
    return redirect(url_for('preview'))

@app.route('/remove_invalid_rows', methods=['POST'])
@login_required
def remove_invalid_rows():
    global df
    violating_data = session.get('violating_rows', [])

    if violating_data:
        violating_df = pd.DataFrame(violating_data)
        df = df.merge(violating_df, how='outer', indicator=True)
        df = df[df['_merge'] == 'left_only'].drop(columns=['_merge'])

    session.pop('violating_rows', None)
    session.pop('validation_rules', None)
    flash("Violating rows removed successfully.", "success")
    return redirect(url_for('preview'))
@app.route('/clear_validation_rules', methods=['POST'])
@login_required
def clear_validation_rules():
    session['validation_rules'] = []
    flash("All validation rules cleared.", "info")
    return redirect(url_for('preview'))





@app.route('/apply_cleaning', methods=['POST'])
@login_required
def apply_cleaning():
    global df, cleaning_report, cleaning_code_lines, cleaned_columns
    if df is None:
        flash('Please upload a dataset first')
        return redirect(url_for('upload_file'))

    col = request.form['column']
    method = request.form['method']
    if col not in df.columns:
        flash(f"Column {col} not found in dataset.")
        return redirect(url_for('preview'))

    # Save current column values before cleaning
    column_versions.setdefault(col, []).append(df[col].copy())
    before_sample = df[col].head(10).copy() if col in df.columns else []

    before = df.shape

    if method == 'drop_na':
        df = drop_na(df, col)
        cleaning_report.append({'issue': f'Missing values in {col}', 'action': 'Dropped rows with NA'})
        cleaning_code_lines.append(f"df = df.dropna(subset=['{col}'])  # drop_na")
    elif method == 'fill_mean':
        df = fill_na_with_mean(df, col)
        cleaning_report.append({'issue': f'Missing values in {col}', 'action': 'Filled NA with mean'})
        cleaning_code_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mean())  # fill_mean")
    elif method == 'fill_median':
        df = fill_na_with_median(df, col)
        cleaning_report.append({'issue': f'Missing values in {col}', 'action': 'Filled NA with median'})
        cleaning_code_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())  # fill_median")
    elif method == 'fill_mode':
        df = fill_na_with_mode(df, col)
        cleaning_report.append({'issue': f'Missing values in {col}', 'action': 'Filled NA with mode'})
        cleaning_code_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mode()[0])  # fill_mode")
    elif method == 'text_clean_basic':
        if col in df.columns:
            df[col] = df[col].apply(clean_text_basic)
            cleaning_report.append({'issue': f'Text cleaning on {col}', 'action': 'Applied basic text cleaning'})
            cleaning_code_lines.append(f"df['{col}'] = df['{col}'].apply(clean_text_basic)  # text_clean_basic")
    elif method == 'remove_outliers':
        before_rows = df.shape[0]
        df = remove_outliers(df, col)
        after_rows = df.shape[0]
        cleaning_report.append({'issue': f'Outliers in {col}', 'action': f'Removed {before_rows - after_rows} outliers'})
        cleaning_code_lines.append(f"df = remove_outliers(df, '{col}')  # remove_outliers")
    elif method == 'convert_dtype':
        df = convert_dtype(df, col)
        cleaning_report.append({'issue': f'Data type issue in {col}', 'action': 'Converted to numeric if possible'})
        cleaning_code_lines.append(f"df = convert_dtype(df, '{col}')  # convert_dtype")
    else:
        flash('Invalid cleaning method selected')
        return redirect(url_for('preview'))
    after_sample = df[col].head(10).copy() if col in df.columns else []
    column_previews[col] = {
    'before': before_sample.tolist(),
    'after': after_sample.tolist()
}


    cleaned_columns.add(col)
    add_version(df)
    flash(f'Cleaning method "{method}" applied on column "{col}"')
    log_action(f'Applied {method} on {col}')
    return redirect(url_for('preview'))

@app.route('/apply_text_cleaning', methods=['POST'])
@login_required
def apply_text_cleaning():
    global df, cleaning_report, cleaning_code_lines
    if df is None:
        flash('Please upload a dataset first')
        return redirect(url_for('upload_file'))

    col = request.form['text_column']
    steps = request.form.getlist('steps')
    if col not in df.columns or not steps:
        flash('Please select column and cleaning steps')
        return redirect(url_for('preview'))

    df[col] = apply_text_cleaning_steps(df[col], steps)
    cleaning_report.append({'issue': f'Text cleaning on {col}', 'action': f'Applied steps: {", ".join(steps)}'})
    cleaning_code_lines.append(f"df['{col}'] = apply_text_cleaning_steps(df['{col}'], {steps})  # advanced_text_cleaning")
    add_version(df)
    flash(f'Text cleaning applied on {col}')
    log_action(f'Applied advanced text cleaning on {col} with steps {steps}')
    return redirect(url_for('preview'))

@app.route('/auto_clean', methods=['POST'])
@login_required
def auto_clean():
    global df, cleaning_report, cleaning_code_lines, cleaned_columns
    if df is None:
        flash('Please upload a dataset first')
        return redirect(url_for('upload_file'))

    for col in df.columns:
        before_sample = df[col].head(10).copy()

        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df = fill_na_with_mean(df, col)
                cleaning_report.append({'issue': f'Missing values in {col}', 'action': 'Filled NA with mean'})
                cleaning_code_lines.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].mean())  # auto_fill_mean")
            else:
                df = drop_na(df, col)
                cleaning_report.append({'issue': f'Missing values in {col}', 'action': 'Dropped rows with NA'})
                cleaning_code_lines.append(f"df = df.dropna(subset=['{col}'])  # auto_drop_na")
            cleaned_columns.add(col)

        if df[col].dtype in [np.float64, np.int64]:
            df = remove_outliers(df, col)
            cleaning_report.append({'issue': f'Outliers in {col}', 'action': 'Removed outliers'})
            cleaning_code_lines.append(f"df = remove_outliers(df, '{col}')  # auto_remove_outliers")
            cleaned_columns.add(col)

        after_sample = df[col].head(10).copy()
        column_previews[col] = {
            'before': before_sample.tolist(),
            'after': after_sample.tolist()
        }

    add_version(df)
    flash('Auto cleaning applied to entire dataset')
    log_action('Auto cleaning applied')
    return redirect(url_for('preview'))


@app.route('/download_cleaned')
@login_required
def download_cleaned():
    global df
    if df is None:
        flash('No dataset available to download')
        return redirect(url_for('upload_file'))
    session['downloaded'] = True

    csv_data = df.to_csv(index=False)

    return send_file(BytesIO(csv_data.encode()), mimetype='text/csv', as_attachment=True, download_name='cleaned_dataset.csv')
    
    

@app.route('/export_pipeline')
@login_required
def export_pipeline():
    code = export_pipeline_code()
    return send_file(BytesIO(code.encode()), mimetype='text/plain', as_attachment=True, download_name='cleaning_pipeline.py')

@app.route('/import_pipeline', methods=['POST'])
@login_required
def import_pipeline():
    global df, cleaning_report, cleaning_code_lines
    if df is None:
        flash('Upload dataset first to apply pipeline')
        return redirect(url_for('upload_file'))

    file = request.files.get('pipeline_file')
    if not file or not file.filename.endswith('.py'):
        flash('Please upload a valid .py pipeline file')
        return redirect(url_for('preview'))

    code = file.read().decode()
    local_vars = {'df': df.copy(), 'clean_text_basic': clean_text_basic, 'remove_outliers': remove_outliers,
                  'fill_na_with_mean': fill_na_with_mean, 'fill_na_with_median': fill_na_with_median,
                  'fill_na_with_mode': fill_na_with_mode, 'drop_na': drop_na, 'convert_dtype': convert_dtype,
                  'apply_text_cleaning_steps': apply_text_cleaning_steps}
    try:
        exec(code, {}, local_vars)
        df_new = local_vars.get('df')
        if df_new is not None:
            df = df_new
            cleaning_report.append({'issue': 'Pipeline imported', 'action': 'Applied imported cleaning pipeline'})
            cleaning_code_lines.append('# Pipeline imported and applied')
            add_version(df)
            flash('Pipeline applied successfully')
        else:
            flash('Pipeline did not produce changes')
    except Exception as e:
        flash(f'Error applying pipeline: {e}')
    return redirect(url_for('preview'))

@app.route('/download_report')
@login_required
def download_report():
    global cleaning_report
    if not cleaning_report:
        flash('No cleaning report available')
        return redirect(url_for('preview'))
    report_str = '\n'.join([f"{entry['issue']}: {entry['action']}" for entry in cleaning_report])
    return send_file(BytesIO(report_str.encode()), mimetype='text/plain', as_attachment=True, download_name='cleaning_report.txt')

@app.route('/show_logs')
@login_required
def show_logs():
    if 'username' not in session:
        flash('Please login to view logs')
        return redirect(url_for('login'))
    user_log_file = f"logs_{session['username']}.txt"
    if not os.path.exists(user_log_file):
        logs = "No logs found."
    else:
        with open(user_log_file, 'r') as f:
            logs = f.read()
    return render_template('logs.html', logs=logs)

@app.route('/undo', methods=['POST'])
@login_required
def undo():
    global df
    if len(data_versions) > 1:
        data_versions.pop()
        df = data_versions[-1]['data'].copy()
        flash("Undo successful: Reverted to previous state")
        log_action("Undo applied")
    else:
        flash("Nothing to undo")
    return redirect(url_for('preview'))

@app.route('/clear_charts')
@login_required
def clear_charts():
    session.pop('chart_history', None)
    flash('All charts have been cleared.')
    return redirect(url_for('preview'))

@app.route('/column_preview/<column>')
@login_required
def column_preview(column):
    if column not in column_previews:
        return jsonify({'error': 'No preview data found'}), 404
    return jsonify(column_previews[column])

@app.route('/revert_column/<column>', methods=['POST'])
@login_required
def revert_column(column):
    global df, column_versions

    if column not in column_versions or not column_versions[column]:
        flash(f"No previous version found for column '{column}'.")
        return redirect(url_for('preview'))

    # Restore the last saved version
    previous = column_versions[column].pop()
    df[column] = previous
    flash(f"Reverted column '{column}' to previous version.")
    return redirect(url_for('preview'))



# ----------------------
# User Auth Routes
# ----------------------

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if os.path.exists(USER_DB):
            existing = pd.read_csv(USER_DB)
            if username in existing['username'].values:
                flash('Username already exists')
                return redirect(url_for('register'))
        hashed = generate_password_hash(password)
        new_user = pd.DataFrame([[username, hashed]], columns=['username', 'password'])
        new_user.to_csv(USER_DB, mode='a', header=not os.path.exists(USER_DB), index=False)
        flash('Registered successfully. Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if os.path.exists(USER_DB):
            users_df = pd.read_csv(USER_DB)
            user_row = users_df[users_df['username'] == username]
            if not user_row.empty and check_password_hash(user_row.iloc[0]['password'], password):
                session['username'] = username
                session['login_success'] = True  # Temporary flag for welcome message
                return redirect(url_for('upload_file'))
        flash('Invalid credentials')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out')
    return redirect(url_for('login'))

# ----------------------
# Run App
# ----------------------
if __name__ == '__main__':
    app.run(debug=True)

