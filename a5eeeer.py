import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, f1_score
import joblib
import base64
from io import BytesIO
import time
import os
from pathlib import Path

st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🚢"
)

st.markdown("""
<style>
.stApp {
    background: #000000;
}

.hero-section {
    position: relative;
    width: 100%;
    height: 70vh;
    margin-bottom: 2rem;
    overflow: hidden;
}

.hero-background {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    filter: brightness(0.9) contrast(1.0);
    animation: fadeIn 2s ease-in;
}

.hero-overlay {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.3);
    z-index: 2;
}

.hero-content {
    position: relative;
    z-index: 3;
    height: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

div.block-container{
    background: rgba(0,0,0,0.85);
    padding: 2rem;
    border-radius: 10px;
    border: 1px solid rgba(230, 200, 115, 0.2);
    margin-top: 2rem;
    position: relative;
    z-index: 10;
}

[data-testid="stHeader"]{
    background: transparent !important;
    border-bottom: none !important;
}

h1,h2,h3,h4,h5{
    color:#e6c873 !important;
    font-weight:800;
}

section[data-testid="stSidebar"]{
    background:#000000;
    color:white;
    border-right: 2px solid #b20000;
}

.stButton>button{
    background: linear-gradient(90deg,#b20000,#e6c873);
    color:white;
    font-weight:bold;
    border-radius:8px;
    padding:10px 20px;
    border:none;
    transition: all 0.3s;
}
.stButton>button:hover{
    transform: scale(1.05);
    box-shadow:0 0 15px #e6c873;
}

.stSuccess > div {
    background: linear-gradient(90deg, #d4af37, #ffd700) !important;
    color: black !important;
    font-weight: 700;
    border-radius: 8px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: rgba(0,0,0,0.5);
    padding: 10px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(230, 200, 115, 0.1);
    border-radius: 5px;
    color: white;
    padding: 10px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #e6c873, #b20000) !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

def get_base64_of_image(image_path):
    try:
        if Path(image_path).exists():
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
        else:
            st.warning(f"⚠️ photo '{image_path}' not found,then use online image.")
            return None
    except Exception as e:
        st.warning(f"⚠️error {e}")
        return None

def display_hero_section():
    image_base64 = get_base64_of_image("large.jpg")
    
    if image_base64:
        background_style = f"url('data:image/jpeg;base64,{image_base64}')"
    else:
        background_style = "url('https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1600px-RMS_Titanic_3.jpg')"
    
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-background" style="background-image: {background_style};"></div>
        <div class="hero-overlay"></div>
        <div class="hero-content">
            <h1 style="
                font-size: 8rem; 
                color: #e6c873 !important; 
                text-shadow: 3px 3px 15px rgba(0,0,0,0.9);
                letter-spacing: 2px;
                font-family: 'Georgia', serif;
                margin: 0;
                animation: titleGlow 3s ease-in-out infinite alternate;
            ">
                TITANIC
            </h1>
        </div>
    </div>
    
    <style>
    @keyframes titleGlow {{
        from {{
            text-shadow: 3px 3px 15px rgba(0,0,0,0.9), 0 0 10px rgba(230, 200, 115, 0.5);
        }}
        to {{
            text-shadow: 3px 3px 20px rgba(0,0,0,1), 0 0 20px rgba(230, 200, 115, 0.8);
        }}
    }}
    </style>
    """, unsafe_allow_html=True)

display_hero_section()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("train.csv")
        return df
    except:
        st.warning("Using sample data as train.csv not found")
        np.random.seed(42)
        n_samples = 891
        
        data = {
            'PassengerId': range(1, n_samples + 1),
            'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
            'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
            'Name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
            'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
            'Age': np.random.normal(29.7, 14.5, n_samples).clip(0, 80),
            'SibSp': np.random.choice([0, 1, 2, 3, 4, 5, 8], n_samples, p=[0.68, 0.23, 0.03, 0.01, 0.01, 0.03, 0.01]),
            'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.09, 0.01, 0.004, 0.002, 0.004]),
            'Fare': np.random.exponential(32.2, n_samples).clip(0, 512),
            'Cabin': np.random.choice(['C101', 'B20', None], n_samples, p=[0.1, 0.1, 0.8]),
            'Embarked': np.random.choice(['C', 'Q', 'S', None], n_samples, p=[0.19, 0.09, 0.72, 0.00]),
        }
        
        df = pd.DataFrame(data)
        df['Embarked'] = df['Embarked'].fillna('S')
        df.loc[df.sample(frac=0.2).index, 'Age'] = np.nan 
        return df

df = load_data()

with st.sidebar:
    st.header("⚙️ Control Panel")
    
    st.subheader("Model Selection")
    model_type = st.selectbox(
        "Choose Model:",
        ["Logistic Regression", "Support Vector Machine (SVM)", "Random Forest"]
    )
    
    if model_type == "Support Vector Machine (SVM)":
        kernel_type = st.selectbox("Kernel Type:", ["rbf", "linear", "poly", "sigmoid"])
        c_value = st.slider("C (Regularization):", 0.1, 10.0, 1.0, 0.1)
    elif model_type == "Random Forest":
        n_estimators = st.slider("Number of Trees:", 10, 200, 100)
        max_depth = st.slider("Max Depth:", 2, 20, 10)
    
    st.subheader("🔧 Preprocessing")
    num_strategy = st.selectbox("Numeric Missing Strategy (Age, Fare):", 
                               ("median", "mean", "most_frequent", "constant"))
    
    cat_strategy = st.selectbox("Categorical Missing Strategy (Embarked, Cabin):",
                               ("most_frequent", "constant (Missing)"))

    test_size = st.slider("Test Size (%):", 10, 40, 20) / 100
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.write(f"Total Passengers: {len(df)}")
    st.write(f"Survival Rate: {(df['Survived'].mean()*100):.1f}%")
    st.write(f"Average Age: {df['Age'].mean():.1f}")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "📈 Analysis", "🤖 Model", "🎯 Predict"])

with tab1:
    st.subheader("📋 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Passengers", len(df))
    with col2:
        survival_rate = df['Survived'].mean() * 100
        st.metric("Survival Rate", f"{survival_rate:.1f}%")
    with col3:
        avg_age = df['Age'].mean()
        st.metric("Average Age", f"{avg_age:.1f}")
    with col4:
        avg_fare = df['Fare'].mean()
        st.metric("Average Fare", f"${avg_fare:.2f}")

    st.subheader("🔍 Data Quality Check")
    
    missing_data = df.isnull().sum().rename('Missing Count')
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    col1_dq, col2_dq = st.columns(2)
    with col1_dq:
        duplicate_rows = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_rows, delta_color="off")
        if duplicate_rows > 0:
             st.warning(f"Found {duplicate_rows} exact duplicate rows.")
        else:
             st.success("No exact duplicate rows found.")
             
    with col2_dq:
        st.markdown("#### Missing Values per Column")
        if not missing_data.empty:
            st.dataframe(missing_data.to_frame(), use_container_width=True)
        else:
            st.success("No missing values found!")

    st.markdown("---")
    
    # ======= NEW HEATMAP FOR MISSING DATA (VALID vs MISSING) =======
    st.subheader("⚠️ Missing Data Heatmap (Valid vs. Missing)")
    
  
    missing_binary_df = df.isna().astype(int).T 
    
    cols = missing_binary_df.columns.astype(str).tolist()
    rows = missing_binary_df.index.tolist()
    
    # Note: df.isna().astype(int) means 1 is Missing, 0 is Valid
    colorscale_miss = [[0, '#e6c873'], [1, 'rgb(178, 0, 0)']] # 0=Gold (Valid), 1=Red (Missing)

    fig_miss = go.Figure(data=go.Heatmap(
        z=missing_binary_df.values,
        x=cols,
        y=rows,
        colorscale=colorscale_miss,
        showscale=False,
        hoverongaps=False
    ))
    
    fig_miss.update_layout(
        title='Data Completeness Check (Green=Valid, Red=Missing)',
        xaxis=dict(showticklabels=False, title='Passenger Samples (Indexed)'), # Hide x-axis labels as they are just row indices
        yaxis=dict(title='Features'),
        height=700
    )
    st.plotly_chart(fig_miss, use_container_width=True)

    st.markdown("---")
    # ======= END MISSING DATA HEATMAP =======
    
    
    # New section for Categorical Missing Value Analysis (Kept for detailed view)
    st.subheader("📊 Categorical Data Analysis (Missing & Unique Values)")
    
    categorical_cols_with_na = df.select_dtypes(include='object').columns[df.select_dtypes(include='object').isnull().any()].tolist()
    
    if categorical_cols_with_na:
        for col in categorical_cols_with_na:
            
            st.markdown(f"**Column: `{col}`**")
            
            total_missing = df[col].isnull().sum()
            unique_values_count = df[col].nunique()
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(f"Missing Values in {col}", total_missing)
            with col_b:
                st.metric(f"Unique Values in {col}", unique_values_count)

            st.markdown(f"***Top 5 Unique Values in `{col}` (excluding NaN):***")
            top_values = df[col].value_counts(dropna=True).head(5)
            st.dataframe(top_values.to_frame(), use_container_width=True)
            
            st.markdown("---")
    else:
        st.success("No categorical columns with missing values found.")


    st.subheader("🔢 Descriptive Statistics (`df.describe()`)")
    
    stat_tab1, stat_tab2 = st.tabs(["Summery Describe", "Data Info"])
    
    with stat_tab1:
        st.markdown("Summery describe for numeric columns (df.describe())")

        st.dataframe(df.describe().T, use_container_width=True)

    with stat_tab2:
        st.markdown("#### Summery for datatype, non-null count, and memory usage (`df.info()`)")
        
        mem_usage = df.memory_usage(index=True, deep=True).sum() / (1024**2)
        st.info(f"**Total Memory Usage:** `{mem_usage:.2f} MB`")
        
        data_info = pd.DataFrame({
            'Dtype (Type)': df.dtypes,
            'Non-Null Count (Not Empty)': df.count(),
            'Missing Count (Empty)': df.isnull().sum()
        })
        st.dataframe(data_info, use_container_width=True)


    st.markdown("---")
    
    st.subheader("Data Preview: Head and Tail")
    
    num_rows = st.slider(
        "Number of Rows for Head/Tail Preview:", 
        min_value=1, 
        max_value=len(df)//10 if len(df) > 10 else len(df), 
        value=5
    )

    col1_dp, col2_dp = st.columns(2)

    with col1_dp:
        st.markdown(f"**First {num_rows} Rows (Head):**")
        st.dataframe(df.head(num_rows), use_container_width=True)

    with col2_dp:
        st.markdown(f"**Last {num_rows} Rows (Tail):**")
        st.dataframe(df.tail(num_rows), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Survival Distribution")
    survived_count = df['Survived'].value_counts().reset_index()
    survived_count.columns = ['Survived', 'Count']
    
    fig = px.pie(survived_count, values='Count', names=['Not Survived', 'Survived'],
                 color=['Not Survived', 'Survived'],
                 color_discrete_map={'Not Survived': '#b20000', 'Survived': '#e6c873'},
                 hole=0.3)
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("📈 Visual Analysis")
    
    # === NEW SECTION: Correlation Heatmap ===
    st.subheader("🔥 Correlation Heatmap (Numeric Features)")
    
    # Select only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=np.number)
    
    # Calculate Correlation Matrix
    corr_matrix = numeric_df.corr().round(2)
    
    # Create the Heatmap using Plotly Graph Objects
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdBu',  # Red-Blue scale is good for correlation (-1 to 1)
        zmin=-1,
        zmax=1,
        text=corr_matrix.values,
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig_corr.update_layout(
        title='Correlation Matrix of Numeric Features',
        xaxis=dict(tickangle=-45),
        yaxis=dict(autorange='reversed'),
        height=500
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    # === END NEW SECTION ===
    
    st.markdown("---")
    
    st.subheader("Distribution and Outliers")
    numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
    
    num_plots = len(numerical_cols)
    num_cols_per_row = 2
    
    for i in range(0, num_plots, num_cols_per_row):
        cols = st.columns(num_cols_per_row)
        for j in range(num_cols_per_row):
            if i + j < num_plots:
                column = numerical_cols[i + j]
                with cols[j]:
                    fig_box = px.box(
                        df, 
                        y=column, 
                        title=f'Boxplot of {column} Distribution', 
                        color_discrete_sequence=['#e6c873'] 
                    )
                    fig_box.update_layout(xaxis_title="") 
                    st.plotly_chart(fig_box, use_container_width=True)


    st.markdown("---")

    fig1 = px.histogram(df, x='Age', color='Survived',
                        color_discrete_map={0: '#b20000', 1: '#e6c873'},
                        nbins=30, barmode='overlay',
                        title="Age Distribution by Survival Status")
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        survival_by_class = df.groupby('Pclass')['Survived'].mean().reset_index()
        fig2 = px.bar(survival_by_class, x='Pclass', y='Survived',
                      color='Survived', color_continuous_scale=['#b20000', '#e6c873'],
                      title="Survival Rate by Passenger Class")
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        survival_by_sex = df.groupby('Sex')['Survived'].mean().reset_index()
        fig3 = px.bar(survival_by_sex, x='Sex', y='Survived',
                      color='Survived', color_continuous_scale=['#b20000', '#e6c873'],
                      title="Survival Rate by Gender")
        st.plotly_chart(fig3, use_container_width=True)

with tab3:
    st.subheader("🤖 Train Machine Learning Model")
    
    if model_type == "Support Vector Machine (SVM)":
        st.info(f"Selected: **{model_type}** with **{kernel_type}** kernel (C={c_value})")
    elif model_type == "Random Forest":
        st.info(f"Selected: **{model_type}** with {n_estimators} trees (max_depth={max_depth})")
    else:
        st.info(f"Selected: **{model_type}**")
    
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'num_imputer' not in st.session_state:
        st.session_state.num_imputer = None
    if 'cat_imputer' not in st.session_state:
        st.session_state.cat_imputer = None
    if 'embarked_mapping' not in st.session_state:
        st.session_state.embarked_mapping = None
    
    if st.button("🚀 Train Model Now", type="primary"):
        with st.spinner("Training model..."):
            df_clean = df.copy()
            
            # ============================================
            # 🔧 PART 1: Handle Missing Values using SimpleImputer
            # ============================================
            
            print("============================================")
            print("STARTING DATA PREPROCESSING AND TRAINING...")
            print("============================================")
            
            # 1. Numerical Features Imputation (Age, Fare)
            numerical_cols = ['Age', 'Fare']
            numerical_cols_exist = [col for col in numerical_cols if col in df_clean.columns]
            
            if numerical_cols_exist:
                if num_strategy == "constant":
                    num_imputer = SimpleImputer(strategy='constant', fill_value=0)
                else:
                    num_imputer = SimpleImputer(strategy=num_strategy)
                
                df_clean[numerical_cols_exist] = num_imputer.fit_transform(
                    df_clean[numerical_cols_exist]
                )
                print(f"✅ Applied '{num_strategy}' imputation to numerical columns: {numerical_cols_exist}")
            
            # 2. Categorical Features Imputation (Embarked, Cabin)
            categorical_cols = ['Embarked', 'Cabin']
            categorical_cols_exist = [col for col in categorical_cols if col in df_clean.columns]
            
            if categorical_cols_exist:
                if cat_strategy == "most_frequent":
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                else: # constant (Missing)
                    cat_imputer = SimpleImputer(strategy='constant', fill_value='Missing')
                
                df_clean[categorical_cols_exist] = cat_imputer.fit_transform(
                    df_clean[categorical_cols_exist]
                )
                print(f"✅ Applied '{cat_strategy}' imputation to categorical columns: {categorical_cols_exist}")
            
            # ============================================
            # 🔄 PART 2: Encode Categorical Variables
            # ============================================
            
            # Encode Sex (binary)
            df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})
            print("✅ Encoded 'Sex' column: male=0, female=1")
            
            # Encode Embarked (multi-class)
            embarked_values = df_clean['Embarked'].unique()
            embarked_mapping = {}
            current_code = 0
            
            # Create mapping for known values + 'Missing' if present
            for val in ['C', 'Q', 'S', 'Missing']:
                if val in embarked_values:
                    embarked_mapping[val] = current_code
                    current_code += 1
            
            # Handle any other unexpected non-NaN values
            for val in embarked_values:
                if val not in embarked_mapping and pd.notna(val):
                    embarked_mapping[val] = current_code
                    current_code += 1
            
            df_clean['Embarked'] = df_clean['Embarked'].map(embarked_mapping)
            print(f"✅ Encoded 'Embarked' column with {len(embarked_mapping)} unique values. Mapping: {embarked_mapping}")
            
            # ============================================
            # 🗑️ PART 3: Drop Unnecessary Columns
            # ============================================
            
            columns_to_drop = ['Cabin', 'Name', 'Ticket', 'PassengerId']
            existing_cols_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
            
            if existing_cols_to_drop:
                df_clean = df_clean.drop(columns=existing_cols_to_drop, errors='ignore')
                print(f"📊 Dropped columns: {existing_cols_to_drop}")
            
            # ============================================
            # 🎯 PART 4: Prepare Features and Target
            # ============================================
            
            features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            available_features = [feat for feat in features if feat in df_clean.columns]
            
            initial_rows = len(df_clean)
            df_clean = df_clean.dropna(subset=available_features)
            final_rows = len(df_clean)
            rows_dropped = initial_rows - final_rows
            
            if rows_dropped > 0:
                print(f"⚠️ Dropped {rows_dropped} rows with remaining missing values after imputation (Initial: {initial_rows}, Final: {final_rows})")
            
            if len(df_clean) == 0:
                print("❌ No data left after cleaning! Check your imputation strategy.")
                st.error("❌ No data left after cleaning! Check your imputation strategy.")
                st.stop()
            
            X = df_clean[available_features].copy()
            y = df_clean['Survived'].copy()
            
            print(f"📊 Using {len(X)} samples with {len(available_features)} features: {available_features}")
            
            # ============================================
            # 🧪 PART 5: Split and Scale Data
            # ============================================
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            print(f"📊 Data split: {len(X_train)} training samples, {len(X_test)} testing samples")
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            print("✅ Features scaled using StandardScaler")
            
            # ============================================
            # 🤖 PART 6: Train Model
            # ============================================
            
            if model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000, random_state=42)
                model_name = "Logistic Regression"
            elif model_type == "Support Vector Machine (SVM)":
                model = SVC(kernel=kernel_type, C=c_value, probability=True, random_state=42)
                model_name = f"SVM ({kernel_type} kernel, C={c_value})"
            else: # Random Forest
                model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    random_state=42
                )
                model_name = f"Random Forest ({n_estimators} trees, max_depth={max_depth})"
            
            model.fit(X_train_scaled, y_train)
            print(f"✅ {model_name} trained successfully!")
            
            # ============================================
            # 💾 PART 7: Store in Session State
            # ============================================
            
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.X_test = X_test_scaled
            st.session_state.y_test = y_test
            st.session_state.model_trained = True
            st.session_state.features = available_features
            
            # Store Imputer and Mapping
            st.session_state.num_imputer = num_imputer if 'num_imputer' in locals() else None
            st.session_state.cat_imputer = cat_imputer if 'cat_imputer' in locals() else None
            st.session_state.embarked_mapping = embarked_mapping
            st.session_state.imputation_strategy = {
                'num_strategy': num_strategy,
                'cat_strategy': cat_strategy
            }
            
            # ============================================
            # 📊 PART 8: Evaluate Model (Display in Streamlit)
            # ============================================
            
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print("============================================")
            print("MODEL EVALUATION RESULTS")
            print(f"Model: {model_name}")
            print(f"Accuracy: {acc*100:.2f}%")
            print(f"F1-Score: {f1:.3f}")
            print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
            print("============================================")

            # Display main success message and key metrics in Streamlit
            st.success(f"✅ Model trained successfully! Accuracy: **{acc*100:.2f}%**, F1-Score: **{f1:.3f}**")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{acc*100:.2f}%")
            
            report = classification_report(y_test, y_pred, output_dict=True)
            with col2:
                precision = report['1']['precision']
                st.metric("Precision", f"{precision:.3f}")
            with col3:
                recall = report['1']['recall']
                st.metric("Recall", f"{recall:.3f}")
            with col4:
                st.metric("F1-Score", f"{f1:.3f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted No', 'Predicted Yes'],
                y=['Actual No', 'Actual Yes'],
                colorscale='Reds',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16}
            ))
            fig_cm.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted Label",
                yaxis_title="True Label"
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            
    
    elif st.session_state.model_trained:
        st.success("✅ Model is already trained and ready for predictions!")
        
        # Display model info
        col1, col2 = st.columns(2)
        with col1:
            if hasattr(st.session_state.model, '__class__'):
                model_name = st.session_state.model.__class__.__name__
                st.info(f"**Model Type:** {model_name}")
        
        with col2:
            if hasattr(st.session_state, 'imputation_strategy'):
                strat = st.session_state.imputation_strategy
                st.info(f"**Imputation:** Num: {strat['num_strategy']}, Cat: {strat['cat_strategy']}")
        
        # Test the model
        y_pred = st.session_state.model.predict(st.session_state.X_test)
        acc = accuracy_score(st.session_state.y_test, y_pred)
        f1 = f1_score(st.session_state.y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", f"{acc*100:.2f}%")
        with col2:
            st.metric("Model F1-Score", f"{f1:.3f}")

with tab4:
    st.subheader("🎯 Predict Survival for New Passenger")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Model' tab!")
        st.info("Go to the **Model** tab and click 'Train Model Now' to get started.")
    else:
        st.success("✅ Model is ready for predictions!")
        
        # Display model and preprocessing info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            if hasattr(st.session_state.model, '__class__'):
                model_name = st.session_state.model.__class__.__name__
                st.info(f"**Model:** {model_name}")
        
        with col_info2:
            if hasattr(st.session_state, 'features'):
                st.info(f"**Features:** {', '.join(st.session_state.features)}")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                pclass = st.selectbox("Passenger Class", [1, 2, 3], key="pclass")
                sex = st.selectbox("Sex", ["male", "female"], key="sex")
                age = st.number_input("Age", min_value=0, max_value=100, value=30, key="age")
            
            with col2:
                sibsp = st.number_input("Siblings/Spouses", min_value=0, max_value=8, value=0, key="sibsp")
                parch = st.number_input("Parents/Children", min_value=0, max_value=6, value=0, key="parch")
                fare = st.number_input("Fare ($)", min_value=0.0, max_value=500.0, value=50.0, key="fare")
                embarked = st.selectbox("Embarked", ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)", "Missing"], key="embarked")
            
            submitted = st.form_submit_button("🔮 Predict Survival", type="primary")
        
        if submitted:
            # Prepare input
            sex_encoded = 0 if sex == "male" else 1
            
            # Handle Embarked encoding using stored mapping
            embarked_mapping = st.session_state.get('embarked_mapping', {})
            
            # Map the selected value (e.g., "Cherbourg (C)" -> "C", or "Missing" -> "Missing")
            embarked_selected = embarked.split(" (")[0] if " (" in embarked else embarked
            
            # If a mapping was learned in training, use it
            if embarked_selected in embarked_mapping:
                embarked_encoded = embarked_mapping[embarked_selected]
                print(f"Prediction Input: Encoded '{embarked}' as {embarked_encoded} using trained mapping.")
            else:
                # Fallback to default mapping if session state is missing
                default_mapping = {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2, "Missing": 3}
                embarked_encoded = default_mapping.get(embarked_selected, 2)
                print(f"Prediction Input: WARNING! Used fallback encoding for '{embarked}': {embarked_encoded}")
            
            # Prepare input array
            input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
            
            # Scale and predict
            input_scaled = st.session_state.scaler.transform(input_data)
            prediction = st.session_state.model.predict(input_scaled)[0]
            
            # Get probability if available
            if hasattr(st.session_state.model, "predict_proba"):
                proba = st.session_state.model.predict_proba(input_scaled)[0]
            else:
                proba = [0.5, 0.5]
            
            st.markdown("---")
            
            # Display prediction result
            if prediction == 1:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(230,200,115,0.2), rgba(230,200,115,0.1));
                    padding: 30px;
                    border-radius: 10px;
                    border: 2px solid #e6c873;
                    text-align: center;
                ">
                    <h2 style="color: #e6c873; font-size: 2.5rem;">🎉 SURVIVED!</h2>
                    <p style="font-size: 1.2rem; color: white;">
                        Based on the trained model, this passenger would <b>likely survive</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(178,0,0,0.2), rgba(178,0,0,0.1));
                    padding: 30px;
                    border-radius: 10px;
                    border: 2px solid #b20000;
                    text-align: center;
                ">
                    <h2 style="color: #b20000; font-size: 2.5rem;">💀 DID NOT SURVIVE</h2>
                    <p style="font-size: 1.2rem; color: white;">
                        Based on the trained model, this passenger would <b>likely not survive</b>.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Show probabilities
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Probability of Survival", f"{proba[1]*100:.1f}%")
            with col2:
                st.metric("Probability of Not Surviving", f"{proba[0]*100:.1f}%")
            
            # Display input values used
            with st.expander("📋 View Input Values Used"):
                input_df = pd.DataFrame({
                    'Feature': st.session_state.features,
                    'Value': [pclass, sex, age, sibsp, parch, f"${fare:.2f}", embarked],
                    'Encoded': [pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]
                })
                st.dataframe(input_df, use_container_width=True)
            
            # Feature importance visualization
            st.subheader("📊 What influenced this prediction?")
            
            if hasattr(st.session_state.model, 'coef_'):
                importance = st.session_state.model.coef_[0]
                features = st.session_state.features
                importance_type = "Coefficient (Linear Model)"
            elif hasattr(st.session_state.model, 'feature_importances_'):
                importance = st.session_state.model.feature_importances_
                features = st.session_state.features
                importance_type = "Feature Importance (Tree-based Model)"
            else:
                importance = np.abs(input_scaled[0])
                features = st.session_state.features
                importance_type = "Scaled Input Magnitude"
            
            st.info(f"**Importance Type:** {importance_type}")
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values(by='Importance', ascending=True)

            fig_imp = px.bar(
                importance_df, 
                x='Importance', 
                y='Feature',
                orientation='h',
                color='Importance',
                color_continuous_scale=['#b20000', '#e6c873'],
                title="Feature Impact on Prediction (Magnitude)"
            )
            fig_imp.update_layout(
                xaxis_title="Importance Score",
                yaxis_title="Feature"
            )
            st.plotly_chart(fig_imp, use_container_width=True)
            
            # Display raw importance values
            with st.expander("📊 View Raw Importance Values"):
                st.dataframe(importance_df.round(4), use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>Titanic Survival Prediction Dashboard • Built with Streamlit • Remembering the 1,517 souls lost in 1912</p>
</div>
""", unsafe_allow_html=True)