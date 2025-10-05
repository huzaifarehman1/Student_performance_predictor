import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.preprocessing import StandardScaler
import joblib
import os

percent = 0.01 #setup by user (for testing data)
neighbours = 5
ignored_column = ["Gender", "Distance_from_Home", "Teacher_Quality", "Tutoring_Sessions"]

# ========== SETTINGS ==========
MODEL_PATH = f"student_prediction(knn)/knn_model{neighbours}.pkl"
SCALER_PATH = f"student_prediction(knn)/scaler{neighbours}.pkl"
DATA_PATH = "student_prediction(knn)/StudentPerformanceFactors.csv"

# ========== TRAIN ONLY IF NOT SAVED ==========
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
    print("⚙️ Training model... please wait")
    data = pd.read_csv(DATA_PATH)
    
    data = data.drop(columns=ignored_column)

    conversion = {"Low":0,"High":2,"Medium":1,"No":0,"Yes":1,"Public":1,"Private":0,
                  'Positive':1,'Negative':2,'Neutral':0,
                  'High School':0,'College':1,'Postgraduate':2, np.nan:1}
    converter = lambda x: conversion[x]

    def grader(n, Forward=True):
        x = [90,80,70,60,50,40,0]
        y = [0,1,2,3,4,5,6]
        if Forward:
            for i in range(len(x)):
                if n >= x[i]:
                    return y[i]
        else:
            for i in range(len(y)):
                if n == y[i]:
                    return (x[i]+10,x[i])

    data["Exam_Score"] = data["Exam_Score"].apply(grader)
    for col in data.columns:
        if col not in ignored_column and isinstance(data[col][2], str):
            data[col] = data[col].apply(converter)

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=percent/100, random_state=42)
    ros = RandomOverSampler(random_state=42)
    x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_resampled)

    model = KNeighborsClassifier(n_neighbors=neighbours, weights='distance')
    model.fit(x_train_scaled, y_train_resampled)

    # ✅ Save once
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(".gitignore","a") as f:
        f.write(f"\n{MODEL_PATH}\n")
        f.write(f"{SCALER_PATH}\n")
    print("✅ Model trained and saved.")
else:
    print("🚀 Loading saved model...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
#print(classification_report(y_test,predictions))
# ======== COLUMNS ========
columns = [
    "Hours_Studied", "Attendance", "Parental_Involvement", "Access_to_Resources",
    "Extracurricular_Activities", "Sleep_Hours", "Previous_Scores", "Motivation_Level",
    "Internet_Access", "Tutoring_Sessions", "Family_Income", "Teacher_Quality",
    "School_Type", "Peer_Influence", "Physical_Activity", "Learning_Disabilities",
    "Parental_Education_Level", "Distance_from_Home", "Gender"
]

# ======== CATEGORY OPTIONS ========
category_options = {
    "Parental_Involvement": ["Low", "Medium", "High"],
    "Access_to_Resources": ["Low", "Medium", "High"],
    "Extracurricular_Activities": ["Yes", "No"],
    "Motivation_Level": ["Low", "Medium", "High"],
    "Internet_Access": ["Yes", "No"],
    "Family_Income": ["Low", "Medium", "High"],
    "School_Type": ["Public", "Private"],
    "Peer_Influence": ["Positive", "Neutral", "Negative"],
    "Learning_Disabilities": ["No", "Yes"],
    "Parental_Education_Level": ["High School", "College", "Postgraduate"]
}

# Example conversion mapping if needed
conversion = {
    "Low": 0, "Medium": 1, "High": 2,
    "Yes": 1, "No": 0,
    "Positive": 2, "Neutral": 1, "Negative": 0,
    "Public": 0, "Private": 1,
    "High School": 0, "College": 1, "Postgraduate": 2
}

# ======== POPUP WINDOW ========
def open_predictor_window(root):
    popup = tk.Toplevel()
    popup.title("🎓 Student Performance Predictor")
    popup.geometry("480x700")
    popup.configure(bg="#f0f4f7")

    # ===== Exit Handler =====
    def close_window():
        if messagebox.askyesno("Exit", "Do you want to cancel and close the program?"):
            popup.destroy()
            root.destroy()  # ✅ completely ends program

    popup.protocol("WM_DELETE_WINDOW", close_window)
    popup.bind("<Escape>", lambda e: close_window())

    # ===== Scrollable Frame =====
    canvas = tk.Canvas(popup, bg="#f0f4f7", highlightthickness=0)
    scrollbar = ttk.Scrollbar(popup, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas, bg="#f0f4f7")

    scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    entries = {}

    # ===== Dynamic Input Creation =====
    for idx, col in enumerate(columns):
        if col in ignored_column:
            continue

        label_text = {
            "Hours_Studied": "Hours Studied per week",
            "Attendance": "Attendance %",
            "Sleep_Hours": "Hours slept per night",
            "Previous_Scores": "Previous exam percentage",
            "Physical_Activity": "Hours of physical activity per week"
        }.get(col, col.replace("_", " "))

        tk.Label(scroll_frame, text=label_text, bg="#f0f4f7",
                 font=("Arial", 10, "bold")).grid(row=idx, column=0, sticky="w", padx=8, pady=4)

        if col in category_options:
            var = tk.StringVar(value=category_options[col][0])
            dropdown = ttk.Combobox(scroll_frame, textvariable=var, values=category_options[col], state="readonly")
            dropdown.grid(row=idx, column=1, padx=8, pady=4)
            entries[col] = var
        else:
            entry = tk.Entry(scroll_frame, width=20)
            entry.grid(row=idx, column=1, padx=8, pady=4)
            entries[col] = entry

    # ===== Predict Function =====
    def predict_exam_score():
        try:
            user_data = []
            for col in columns:
                if col in ignored_column:
                    continue
                val = entries[col].get().strip()
                if val == "":
                    messagebox.showwarning("Missing Input", f"Please enter a value for '{col}'.")
                    return
                val = conversion.get(val, val)
                val = float(val)
                user_data.append(val)

            test_df = pd.DataFrame([user_data], columns=[c for c in columns if c not in ignored_column])
            test_scaled = scaler.transform(test_df)
            pred = model.predict(test_scaled)[0]
            def grader(n, Forward=True):
                x = [90,80,70,60,50,40,0]
                y = [0,1,2,3,4,5,6]
                if Forward:
                    for i in range(len(x)):
                        if n >= x[i]:
                            return y[i]
                else:
                    for i in range(len(y)):
                        if n == y[i]:
                            return (x[i]+10,x[i])
            score_range = grader(pred, Forward=False)
            messagebox.showinfo(
                "Prediction Result",
                f"🎯 Predicted Grade Level: {pred}\n"
                f"📈 Expected Exam Score Range: {score_range[1]}–{score_range[0]}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"⚠️ {e}")

    # ===== Buttons =====
    tk.Button(scroll_frame, text="Predict", command=predict_exam_score,
              bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
              padx=10, pady=5).grid(row=len(columns)+1, column=0, columnspan=2, pady=15)

    tk.Button(scroll_frame, text="Cancel & Exit", command=close_window,
              bg="#d9534f", fg="white", font=("Arial", 10, "bold"),
              padx=10, pady=5).grid(row=len(columns)+2, column=0, columnspan=2, pady=5)

    # ===== Layout =====
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

# ======== MAIN ========
root = tk.Tk()
root.withdraw()  # hide main window
open_predictor_window(root)
root.mainloop()
