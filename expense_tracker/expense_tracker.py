# Expense Tracker with Machine Learning
# Tech stack: pandas, numpy, matplotlib, seaborn, scikit-learn
#
# ML overview:
#   Category prediction uses Naive Bayes. It learns which words appear
#   most often in each category (Food, Travel, etc.) and uses that to
#   classify new descriptions. Simple but works well for short text.
#
#   Spending forecast uses Linear Regression. It fits a straight line
#   through your monthly totals and extends it one month forward.

import os
import csv
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")


# --- Configuration ---

CSV_FILE = "expenses.csv"
DATE_FORMAT = "%Y-%m-%d"

CATEGORIES = [
    "Food",
    "Travel",
    "Bills",
    "Entertainment",
    "Healthcare",
    "Shopping",
    "Education",
    "Other",
]

PALETTE = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
]


# --- Data handling ---

def initialize_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Date", "Category", "Amount", "Description"])
        print(f"Created new file: {CSV_FILE}")


def load_expenses():
    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            return df
        df["Date"] = pd.to_datetime(df["Date"], format=DATE_FORMAT)
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0)
        return df
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["ID", "Date", "Category", "Amount", "Description"])


def save_expenses(df):
    df["Date"] = df["Date"].dt.strftime(DATE_FORMAT)
    df.to_csv(CSV_FILE, index=False)


def get_next_id(df):
    if df.empty or "ID" not in df.columns:
        return 1
    return int(df["ID"].max()) + 1


def add_expense():
    print("\n  Add New Expense")

    date_str = input("  Date (YYYY-MM-DD) [Enter for today]: ").strip()
    if not date_str:
        date_str = datetime.today().strftime(DATE_FORMAT)
    try:
        datetime.strptime(date_str, DATE_FORMAT)
    except ValueError:
        print("  Invalid date format.")
        return

    print("\n  Categories:")
    for i, cat in enumerate(CATEGORIES, 1):
        print(f"    {i}. {cat}")
    cat_input = input("  Choose number or type name: ").strip()
    if cat_input.isdigit() and 1 <= int(cat_input) <= len(CATEGORIES):
        category = CATEGORIES[int(cat_input) - 1]
    elif cat_input.title() in CATEGORIES:
        category = cat_input.title()
    else:
        print("  Invalid category.")
        return

    amount_str = input("  Amount: ").strip()
    try:
        amount = float(amount_str)
        if amount <= 0:
            raise ValueError
    except ValueError:
        print("  Invalid amount. Enter a positive number.")
        return

    description = input("  Description: ").strip()
    if not description:
        print("  Description cannot be empty.")
        return

    df = load_expenses()
    new_id = get_next_id(df)
    new_row = pd.DataFrame([{
        "ID": new_id,
        "Date": date_str,
        "Category": category,
        "Amount": amount,
        "Description": description,
    }])
    df["Date"] = df["Date"].dt.strftime(DATE_FORMAT)
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"\n  Added expense #{new_id}: {category} - {amount:.2f} ({description})")


def view_expenses():
    df = load_expenses()
    if df.empty:
        print("\n  No expenses recorded yet.")
        return

    print("\n  All Expenses\n")
    print(f"  {'ID':<5} {'Date':<12} {'Category':<15} {'Amount':>10}  Description")
    print("  " + "-" * 65)
    for _, row in df.iterrows():
        date_str = row["Date"].strftime(DATE_FORMAT)
        print(f"  {int(row['ID']):<5} {date_str:<12} {row['Category']:<15} "
              f"{row['Amount']:>10,.2f}  {row['Description']}")

    print("  " + "-" * 65)
    print(f"  {'TOTAL':<33} {df['Amount'].sum():>10,.2f}")

    print("\n  Category Summary\n")
    summary = df.groupby("Category")["Amount"].agg(["sum", "count"])
    summary.columns = ["Total", "Count"]
    summary = summary.sort_values("Total", ascending=False)
    for cat, row in summary.iterrows():
        print(f"  {cat:<15}  {row['Total']:>10,.2f}   ({int(row['Count'])} entries)")


def edit_expense():
    df = load_expenses()
    if df.empty:
        print("\n  No expenses to edit.")
        return

    try:
        expense_id = int(input("\n  Expense ID to edit: ").strip())
    except ValueError:
        print("  Invalid ID.")
        return

    if expense_id not in df["ID"].values:
        print(f"  Expense ID {expense_id} not found.")
        return

    idx = df[df["ID"] == expense_id].index[0]
    row = df.loc[idx]
    print(f"\n  Editing: {row['Category']} - {row['Amount']} - {row['Description']}")
    print("  Press Enter to keep the current value.\n")

    new_date = input(f"  Date [{row['Date'].strftime(DATE_FORMAT)}]: ").strip()
    if new_date:
        try:
            datetime.strptime(new_date, DATE_FORMAT)
            df.at[idx, "Date"] = new_date
        except ValueError:
            print("  Invalid date. Keeping original.")

    print(f"\n  Categories: {', '.join(CATEGORIES)}")
    new_cat = input(f"  Category [{row['Category']}]: ").strip().title()
    if new_cat and new_cat in CATEGORIES:
        df.at[idx, "Category"] = new_cat
    elif new_cat:
        print("  Invalid category. Keeping original.")

    new_amount = input(f"  Amount [{row['Amount']}]: ").strip()
    if new_amount:
        try:
            df.at[idx, "Amount"] = float(new_amount)
        except ValueError:
            print("  Invalid amount. Keeping original.")

    new_desc = input(f"  Description [{row['Description']}]: ").strip()
    if new_desc:
        df.at[idx, "Description"] = new_desc

    save_expenses(df)
    print(f"\n  Expense #{expense_id} updated.")


def delete_expense():
    df = load_expenses()
    if df.empty:
        print("\n  No expenses to delete.")
        return

    try:
        expense_id = int(input("\n  Expense ID to delete: ").strip())
    except ValueError:
        print("  Invalid ID.")
        return

    if expense_id not in df["ID"].values:
        print(f"  Expense ID {expense_id} not found.")
        return

    row = df[df["ID"] == expense_id].iloc[0]
    print(f"\n  Will delete: {row['Category']} - {row['Amount']} - {row['Description']}")
    confirm = input("  Are you sure? (yes/no): ").strip().lower()
    if confirm == "yes":
        df = df[df["ID"] != expense_id]
        save_expenses(df)
        print(f"  Expense #{expense_id} deleted.")
    else:
        print("  Deletion cancelled.")


# --- Analysis and visualization ---

def monthly_summary(df):
    df = df.copy()
    df["Month"] = df["Date"].dt.to_period("M")
    monthly = df.groupby("Month")["Amount"].sum().reset_index()
    monthly.columns = ["Month", "Total"]
    print("\n  Monthly Summary\n")
    for _, row in monthly.iterrows():
        print(f"  {str(row['Month']):<10}  {row['Total']:>10,.2f}")
    return monthly


def weekly_summary(df):
    df = df.copy()
    df["Week"] = df["Date"].dt.to_period("W")
    weekly = df.groupby("Week")["Amount"].sum().reset_index()
    weekly.columns = ["Week", "Total"]
    print("\n  Weekly Summary\n")
    for _, row in weekly.iterrows():
        print(f"  {str(row['Week']):<25}  {row['Total']:>10,.2f}")
    return weekly


def plot_charts(df):
    if df.empty:
        print("  No data to visualize.")
        return

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Expense Analysis", fontsize=16, fontweight="bold", y=1.01)

    # Spending by category
    cat_totals = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
    axes[0, 0].bar(cat_totals.index, cat_totals.values,
                   color=PALETTE[:len(cat_totals)], edgecolor="white")
    axes[0, 0].set_title("Spending by Category", fontweight="bold")
    axes[0, 0].set_xlabel("Category")
    axes[0, 0].set_ylabel("Total Amount")
    axes[0, 0].tick_params(axis="x", rotation=30)

    # Category share as a pie chart
    axes[0, 1].pie(
        cat_totals.values,
        labels=cat_totals.index,
        autopct="%1.1f%%",
        colors=PALETTE[:len(cat_totals)],
        startangle=140,
        pctdistance=0.82,
    )
    axes[0, 1].set_title("Category Share", fontweight="bold")

    # Monthly trend
    df_copy = df.copy()
    df_copy["Month"] = df_copy["Date"].dt.to_period("M")
    monthly = df_copy.groupby("Month")["Amount"].sum()
    month_labels = [str(m) for m in monthly.index]
    axes[1, 0].plot(month_labels, monthly.values, marker="o",
                    color="#4ECDC4", linewidth=2.5, markersize=7)
    axes[1, 0].fill_between(month_labels, monthly.values, alpha=0.15, color="#4ECDC4")
    axes[1, 0].set_title("Monthly Trend", fontweight="bold")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Total Amount")
    axes[1, 0].tick_params(axis="x", rotation=30)

    # Top 5 largest expenses
    top5 = df.nlargest(5, "Amount")[["Description", "Amount"]].copy()
    top5["Description"] = top5["Description"].str[:20]
    axes[1, 1].barh(top5["Description"], top5["Amount"],
                    color=PALETTE[:5], edgecolor="white")
    axes[1, 1].set_title("Top 5 Expenses", fontweight="bold")
    axes[1, 1].set_xlabel("Amount")
    axes[1, 1].invert_yaxis()

    plt.tight_layout()
    plt.savefig("expense_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Charts saved to expense_analysis.png")


def analyze_data():
    df = load_expenses()
    if df.empty:
        print("\n  No data available. Add some expenses first.")
        return

    print("\n  Analysis Options")
    print("  1. Monthly Summary")
    print("  2. Weekly Summary")
    print("  3. View Charts")
    print("  4. All")

    choice = input("\n  Choose (1-4): ").strip()
    if choice in ("1", "4"):
        monthly_summary(df)
    if choice in ("2", "4"):
        weekly_summary(df)
    if choice in ("3", "4"):
        plot_charts(df)


# --- Machine learning ---

def train_category_classifier(df):
    # Needs enough data and category variety to be useful
    if df.shape[0] < 10:
        print("\n  Need at least 10 expenses to train the model.")
        print("  Load sample data (option 9) to get started.")
        return None, None, None

    X = df["Description"].astype(str)
    y = df["Category"].astype(str)

    if y.nunique() < 2:
        print("\n  Need expenses from at least 2 different categories.")
        return None, None, None

    # Convert descriptions to word-count vectors (bag of words approach)
    vectorizer = CountVectorizer(lowercase=True, stop_words="english")
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    # alpha=1.0 is standard Laplace smoothing, prevents zero-probability issues
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, vectorizer, accuracy


def predict_category():
    df = load_expenses()
    model, vectorizer, accuracy = train_category_classifier(df)
    if model is None:
        return

    print(f"\n  Model trained. Test accuracy: {accuracy * 100:.1f}%")
    print(f"  Correct on {accuracy * 100:.0f}% of descriptions it had not seen before.\n")

    while True:
        desc = input("  Enter a description to classify (or 'done' to stop): ").strip()
        if desc.lower() == "done":
            break
        if not desc:
            continue

        X_new = vectorizer.transform([desc])
        predicted = model.predict(X_new)[0]
        confidence = max(model.predict_proba(X_new)[0]) * 100
        print(f"  Predicted: {predicted}  (confidence: {confidence:.1f}%)")


def predict_future_expenses():
    # Assign each month a number (1, 2, 3...), fit a line through monthly
    # totals, then read off the value at position N+1 as the forecast.
    df = load_expenses()
    if df.empty:
        print("\n  No data to forecast.")
        return

    df_copy = df.copy()
    df_copy["Month"] = df_copy["Date"].dt.to_period("M")
    monthly = df_copy.groupby("Month")["Amount"].sum().reset_index()

    if len(monthly) < 3:
        print("\n  Need at least 3 months of data for a meaningful forecast.")
        return

    X = np.arange(1, len(monthly) + 1).reshape(-1, 1)
    y = monthly["Amount"].values

    model = LinearRegression()
    model.fit(X, y)

    r2 = model.score(X, y)
    prediction = max(model.predict(np.array([[len(monthly) + 1]]))[0], 0)
    next_month_label = (monthly["Month"].iloc[-1] + 1).strftime("%B %Y")

    print(f"\n  Spending Forecast\n")
    print(f"  R-squared: {r2:.2f}  (1.0 = line fits perfectly, 0.0 = no pattern)")
    print(f"  Monthly trend: {model.coef_[0]:+.2f} per month")
    print(f"\n  Forecast for {next_month_label}: {prediction:,.2f}")

    if r2 < 0.5:
        print("\n  Note: Low R-squared means spending is irregular.")
        print("  Treat this as a rough estimate only.")

    plt.figure(figsize=(9, 5))
    month_labels = [str(m) for m in monthly["Month"]]
    plt.scatter(X, y, color="#FF6B6B", s=80, zorder=5, label="Actual")
    line_x = np.arange(1, len(monthly) + 2).reshape(-1, 1)
    plt.plot(line_x, model.predict(line_x), color="#4ECDC4",
             linewidth=2, linestyle="--", label="Trend")
    plt.scatter([len(monthly) + 1], [prediction],
                color="#FFEAA7", s=120, zorder=6, marker="*", label="Forecast")
    plt.xticks(list(range(1, len(monthly) + 2)),
               month_labels + [next_month_label], rotation=30)
    plt.title("Monthly Spending with Forecast", fontweight="bold")
    plt.xlabel("Month")
    plt.ylabel("Total Amount")
    plt.legend()
    plt.tight_layout()
    plt.savefig("expense_forecast.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Forecast chart saved to expense_forecast.png")


def show_model_report():
    df = load_expenses()
    model, vectorizer, accuracy = train_category_classifier(df)
    if model is None:
        return

    X = df["Description"].astype(str)
    y = df["Category"].astype(str)
    y_pred = model.predict(vectorizer.transform(X))

    print(f"\n  Overall Accuracy: {accuracy * 100:.1f}%\n")
    print("  Detailed Report\n")
    print(classification_report(y, y_pred, zero_division=0))
    print("  Precision: of all predictions for category X, how many were correct?")
    print("  Recall:    of all actual X expenses, how many did the model catch?")
    print("  F1-score:  balance between precision and recall (higher is better)")


# --- Sample data ---

SAMPLE_DATA = [
    ("2024-10-03", "Food",          450,  "pizza dinner with friends"),
    ("2024-10-05", "Bills",        1200,  "electricity bill payment"),
    ("2024-10-07", "Travel",        800,  "uber cab to airport"),
    ("2024-10-10", "Food",          180,  "biryani lunch"),
    ("2024-10-12", "Entertainment", 350,  "netflix subscription"),
    ("2024-10-14", "Shopping",      999,  "new running shoes"),
    ("2024-10-16", "Food",          220,  "grocery vegetables"),
    ("2024-10-18", "Healthcare",    600,  "doctor consultation fee"),
    ("2024-10-20", "Travel",        300,  "metro card recharge"),
    ("2024-10-22", "Education",    1500,  "online python course"),
    ("2024-10-25", "Food",          400,  "restaurant dinner"),
    ("2024-10-28", "Bills",        3500,  "internet broadband plan"),
    ("2024-11-01", "Food",          260,  "coffee and snacks"),
    ("2024-11-03", "Shopping",     2500,  "winter jacket clothing"),
    ("2024-11-05", "Travel",       1100,  "train ticket booking"),
    ("2024-11-08", "Food",          320,  "dosa breakfast"),
    ("2024-11-10", "Entertainment", 700,  "movie theatre tickets"),
    ("2024-11-12", "Healthcare",    450,  "pharmacy medicine"),
    ("2024-11-15", "Bills",        1100,  "phone recharge plan"),
    ("2024-11-18", "Education",     800,  "study books purchase"),
    ("2024-11-20", "Food",          510,  "birthday party food"),
    ("2024-11-22", "Travel",        650,  "fuel petrol car"),
    ("2024-11-25", "Shopping",     1800,  "laptop accessories"),
    ("2024-11-28", "Food",          190,  "chai samosa evening snack"),
    ("2024-12-01", "Bills",        2000,  "rent utility payment"),
    ("2024-12-04", "Food",          370,  "paneer dinner"),
    ("2024-12-06", "Entertainment", 500,  "spotify premium music"),
    ("2024-12-08", "Travel",        420,  "auto rickshaw rides"),
    ("2024-12-10", "Healthcare",    900,  "blood test lab"),
    ("2024-12-12", "Food",          280,  "vegetable fruits grocery"),
    ("2024-12-15", "Shopping",     3200,  "smartphone accessories"),
    ("2024-12-18", "Education",    1200,  "workshop registration fee"),
    ("2024-12-20", "Food",          560,  "family restaurant lunch"),
    ("2024-12-22", "Travel",        750,  "bus interstate journey"),
    ("2024-12-25", "Entertainment", 400,  "board game purchase"),
    ("2024-12-28", "Bills",         800,  "gym membership renewal"),
    ("2025-01-02", "Food",          300,  "pizza order online"),
    ("2025-01-05", "Shopping",     1500,  "clothes festival sale"),
    ("2025-01-08", "Healthcare",    350,  "vitamins supplement"),
    ("2025-01-10", "Travel",        200,  "bus ticket local"),
    ("2025-01-12", "Food",          480,  "chole bhature lunch"),
    ("2025-01-15", "Bills",        1500,  "water electricity bill"),
    ("2025-01-18", "Entertainment", 600,  "concert event tickets"),
    ("2025-01-20", "Education",    2000,  "certification exam fee"),
    ("2025-01-22", "Food",          230,  "juice smoothie breakfast"),
    ("2025-01-25", "Travel",        900,  "cab airport transfer"),
    ("2025-01-28", "Shopping",      750,  "kitchen utensils"),
]


def load_sample_data():
    df = load_expenses()
    existing_count = len(df)
    next_id = get_next_id(df)

    new_rows = []
    for date, category, amount, description in SAMPLE_DATA:
        new_rows.append({
            "ID": next_id,
            "Date": date,
            "Category": category,
            "Amount": amount,
            "Description": description,
        })
        next_id += 1

    new_df = pd.DataFrame(new_rows)
    df["Date"] = df["Date"].dt.strftime(DATE_FORMAT)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"\n  Loaded {len(new_rows)} sample expenses.")
    print(f"  Total expenses now: {existing_count + len(new_rows)}")


# --- Menu ---

MENU = """
  Expense Tracker
  ---------------
  1. Add Expense
  2. View All Expenses
  3. Edit an Expense
  4. Delete an Expense
  5. Analyze Data
  6. ML: Predict Category
  7. ML: Forecast Future Spending
  8. ML: Show Model Report
  9. Load Sample Data
  0. Exit
"""

ACTIONS = {
    "1": add_expense,
    "2": view_expenses,
    "3": edit_expense,
    "4": delete_expense,
    "5": analyze_data,
    "6": predict_category,
    "7": predict_future_expenses,
    "8": show_model_report,
    "9": load_sample_data,
}


def main():
    initialize_csv()
    print("\n  Expense Tracker with ML")
    print("  Start with option 9 to load sample data.\n")

    while True:
        print(MENU)
        choice = input("  Choice: ").strip()

        if choice == "0":
            print("\n  Exiting.\n")
            sys.exit(0)
        elif choice in ACTIONS:
            try:
                ACTIONS[choice]()
            except KeyboardInterrupt:
                print("\n  Cancelled.")
        else:
            print("  Invalid choice. Enter a number from 0 to 9.")

        input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()
