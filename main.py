from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import pandas as pd
import sqlite3
import io
import json
import os
import pickle
from collections import defaultdict
from contextlib import contextmanager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DB_PATH = "finance.db"

class SmartCategorizer:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 3),
                max_features=2000,
                min_df=1,
                max_df=0.8,
                sublinear_tf=True
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        self.is_trained = False
        self.model_path = 'categorizer_model.pkl'
        self.seed_data_path = 'seed_training_data.json'
        
        if os.path.exists(self.model_path):
            try:
                self.load_model()
            except:
                pass
    
    def get_seed_training_data(self):
        """Generate diverse seed training examples from common patterns."""
        seed_examples = {
            'groceries': [
                'whole foods market', 'trader joes grocery', 'safeway store',
                'walmart supercenter', 'target groceries', 'costco wholesale',
                'albertsons market', 'kroger supermarket', 'publix super market',
                'aldi food store', 'food lion', 'wegmans market', 'heb grocery',
                'sprouts farmers market', 'natural grocers', 'smart final',
                'food 4 less', 'giant eagle', 'stop shop', 'vons',
                'raleys supermarket', 'winco foods', 'grocery outlet'
            ],
            'dining': [
                'starbucks coffee', 'mcdonalds restaurant', 'chipotle mexican grill',
                'panera bread cafe', 'subway sandwich', 'taco bell', 'wendys',
                'burger king', 'chick fil a', 'five guys burgers', 'shake shack',
                'olive garden', 'red lobster', 'outback steakhouse', 'applebees',
                'cheesecake factory', 'buffalo wild wings', 'panda express',
                'dominos pizza', 'pizza hut', 'papa johns', 'little caesars',
                'dunkin donuts', 'krispy kreme', 'in n out burger',
                'jersey mikes subs', 'jimmy johns', 'qdoba mexican',
                'sonic drive in', 'arbys roast beef', 'popeyes chicken',
                'kfc fried chicken', 'long john silvers', 'dairy queen',
                'doordash delivery', 'uber eats order', 'grubhub food',
                'postmates delivery', 'seamless order', 'caviar delivery',
                'local restaurant', 'cafe bistro', 'coffee shop'
            ],
            'transportation': [
                'uber ride', 'lyft driver', 'shell gas station', 'chevron fuel',
                'exxon mobil', 'bp gas', 'speedway', 'circle k', '76 gas station',
                'arco ampm', 'valero', 'marathon', 'sunoco', 'citgo',
                'racetrac', 'wawa', 'sheetz', 'quiktrip', 'loves travel',
                'parking garage', 'parking meter', 'parking lot fee',
                'metro transit', 'subway fare', 'bus pass', 'train ticket',
                'commuter rail', 'public transportation', 'toll road',
                'car wash', 'auto service', 'oil change', 'tire rotation'
            ],
            'utilities': [
                'electric company', 'power bill', 'gas electric',
                'water utility', 'sewer service', 'waste management',
                'internet service', 'comcast xfinity', 'spectrum cable',
                'verizon fios', 'att fiber', 'cox communications',
                'phone bill', 'mobile service', 't mobile', 'sprint',
                'boost mobile', 'cricket wireless', 'straight talk',
                'cable television', 'satellite tv', 'dish network', 'directv'
            ],
            'entertainment': [
                'netflix subscription', 'spotify premium', 'hulu streaming',
                'disney plus', 'amazon prime video', 'hbo max', 'apple tv',
                'youtube premium', 'paramount plus', 'peacock', 'discovery',
                'amc theaters', 'regal cinema', 'cinemark', 'movie theater',
                'imax', 'alamo drafthouse', 'showcase cinema',
                'playstation store', 'xbox live', 'nintendo eshop', 'steam games',
                'epic games', 'battle net', 'origin', 'gog games',
                'spotify music', 'apple music', 'tidal', 'pandora',
                'audible audiobooks', 'kindle unlimited', 'scribd',
                'crunchyroll anime', 'funimation', 'twitch subscription',
                'patreon support', 'discord nitro', 'adobe creative cloud'
            ],
            'shopping': [
                'amazon purchase', 'amazon com', 'best buy electronics',
                'target store', 'walmart shopping', 'macys department',
                'nordstrom', 'kohls store', 'jcpenney', 'dillards',
                'apple store', 'microsoft store', 'gamestop', 'barnes noble',
                'bed bath beyond', 'home depot', 'lowes hardware',
                'ikea furniture', 'wayfair home', 'overstock',
                'etsy handmade', 'ebay purchase', 'newegg tech',
                'micro center', 'staples office', 'office depot',
                'petco pet supplies', 'petsmart', 'chewy pet food',
                'ulta beauty', 'sephora cosmetics', 'sally beauty',
                'ross dress less', 'tj maxx', 'marshalls', 'burlington'
            ],
            'health': [
                'cvs pharmacy', 'walgreens drugstore', 'rite aid',
                'walmart pharmacy', 'target pharmacy', 'kroger pharmacy',
                'doctor visit', 'physician office', 'medical center',
                'hospital emergency', 'urgent care', 'clinic',
                'dental office', 'dentist appointment', 'orthodontist',
                'vision center', 'eye doctor', 'optometrist',
                'physical therapy', 'chiropractor', 'massage therapy',
                'mental health', 'therapist counseling', 'psychiatrist',
                'lab work', 'blood test', 'imaging center', 'radiology',
                'specialist doctor', 'cardiologist', 'dermatologist',
                'prescription medication', 'pharmacy copay'
            ],
            'income': [
                'payroll deposit', 'direct deposit', 'salary payment',
                'wages earned', 'paycheck', 'bonus payment', 'commission',
                'freelance payment', 'contract work', 'consulting fee',
                'dividend income', 'investment dividend', 'stock dividend',
                'interest payment', 'interest earned', 'savings interest',
                'tax refund', 'federal refund', 'state refund',
                'reimbursement', 'expense reimbursement', 'refund credit',
                'cash back reward', 'rebate', 'gift received',
                'venmo received', 'paypal received', 'zelle transfer in',
                'side hustle income', 'gig economy', 'uber driver pay',
                'etsy sales', 'rental income', 'royalty payment'
            ]
        }
        
        return seed_examples
    
    def train(self, descriptions, categories):
        """Train on existing categorized transactions."""
        if len(descriptions) < 5:
            return False
        
        self.model.fit(descriptions, categories)
        self.is_trained = True
        self.save_model()
        return True
    
    def predict(self, descriptions):
        """Predict categories for new transactions."""
        if not self.is_trained:
            return ['other'] * len(descriptions)
        
        try:
            return self.model.predict(descriptions).tolist()
        except:
            return ['other'] * len(descriptions)
    
    def predict_single(self, description):
        """Predict category for a single transaction."""
        return self.predict([description])[0]
    
    def predict_with_confidence(self, description):
        """Predict category with confidence score."""
        if not self.is_trained:
            return 'other', 0.0
        
        try:
            proba = self.model.predict_proba([description])[0]
            category = self.model.classes_[proba.argmax()]
            confidence = proba.max()
            return category, confidence
        except:
            return 'other', 0.0
    
    def save_model(self):
        """Save trained model to disk."""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load trained model from disk."""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
            self.is_trained = True

# Initialize categorizer
categorizer = SmartCategorizer()

# 50-30-20 Rule Categorization
NEEDS_CATEGORIES = ['groceries', 'utilities', 'transportation', 'health']
WANTS_CATEGORIES = ['dining', 'entertainment', 'shopping', 'other']
SAVINGS_CATEGORIES = []

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db() as conn:
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                description TEXT NOT NULL,
                amount REAL NOT NULL,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()

def categorize_transaction(description: str) -> str:
    return categorizer.predict_single(description)

# Models
class Transaction(BaseModel):
    date: str
    description: str
    amount: float
    category: Optional[str] = None

class BudgetSuggestion(BaseModel):
    category: str
    current_spending: float
    suggested_budget: float
    status: str

# Routes
@app.on_event("startup")
async def startup():
    init_db()
    
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT description, category FROM transactions WHERE category != 'other'")
            data = cur.fetchall()
            
            if len(data) >= 5:
                descriptions = [row['description'] for row in data]
                categories = [row['category'] for row in data]
                
                if categorizer.train(descriptions, categories):
                    print(f"ML Model trained on {len(data)} transactions!")
                else:
                    print("Not enough data to train model")
            else:
                seed_data = categorizer.get_seed_training_data()
                all_descriptions = []
                all_categories = []
                
                for category, examples in seed_data.items():
                    all_descriptions.extend(examples)
                    all_categories.extend([category] * len(examples))
                
                if categorizer.train(all_descriptions, all_categories):
                    print(f"ML Model initialized with {len(all_descriptions)} diverse training examples")
                else:
                    print("Failed to initialize model")
    except Exception as e:
        print(f"Error during startup: {e}")
    
    print("Database initialized successfully!")

@app.get("/")
async def root():
    return {"message": "Personal Finance Dashboard API (SQLite)", "database": DB_PATH}

@app.post("/upload")
async def upload_transactions(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        date_col = next((col for col in df.columns if 'date' in col.lower()), None)
        desc_col = next((col for col in df.columns if 'desc' in col.lower() or 'name' in col.lower()), None)
        amount_col = next((col for col in df.columns if 'amount' in col.lower() or 'price' in col.lower()), None)
        
        if not all([date_col, desc_col, amount_col]):
            raise HTTPException(status_code=400, detail="CSV must contain date, description, and amount columns")
        
        descriptions = df[desc_col].tolist()
        categories = categorizer.predict(descriptions)
        
        with get_db() as conn:
            cur = conn.cursor()
            
            inserted = 0
            for idx, row in df.iterrows():
                try:
                    date = pd.to_datetime(row[date_col]).date()
                    description = str(row[desc_col])
                    amount = float(row[amount_col])
                    category = categories[idx]
                    
                    cur.execute(
                        "INSERT INTO transactions (date, description, amount, category) VALUES (?, ?, ?, ?)",
                        (date, description, amount, category)
                    )
                    inserted += 1
                except Exception as e:
                    continue
            
            conn.commit()
            
            cur.execute("SELECT description, category FROM transactions WHERE category != 'other'")
            all_data = cur.fetchall()
            
            if len(all_data) >= 5:
                all_descriptions = [row['description'] for row in all_data]
                all_categories = [row['category'] for row in all_data]
                
                if categorizer.train(all_descriptions, all_categories):
                    print(f"Model retrained on {len(all_data)} transactions")
        
        return {"message": f"Successfully uploaded {inserted} transactions. Model updated!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/transactions")
async def get_transactions(limit: int = 100):
    with get_db() as conn:
        cur = conn.cursor()
        
        cur.execute(
            "SELECT * FROM transactions ORDER BY date DESC LIMIT ?",
            (limit,)
        )
        
        transactions = [dict(row) for row in cur.fetchall()]
    
    return transactions

@app.get("/spending-by-category")
async def spending_by_category():
    with get_db() as conn:
        cur = conn.cursor()
        
        cur.execute("""
            SELECT category, SUM(amount) as total
            FROM transactions
            WHERE amount < 0
            GROUP BY category
            ORDER BY total
        """)
        
        data = [dict(row) for row in cur.fetchall()]
    
    return data

@app.get("/monthly-spending")
async def monthly_spending():
    with get_db() as conn:
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                strftime('%Y-%m', date) as month,
                category,
                SUM(amount) as total
            FROM transactions
            WHERE amount < 0
            GROUP BY month, category
            ORDER BY month DESC, category
        """)
        
        data = [dict(row) for row in cur.fetchall()]
    
    return data

@app.get("/budget-suggestions")
async def budget_suggestions():
    with get_db() as conn:
        cur = conn.cursor()
        
        cur.execute("""
            SELECT AVG(monthly_income) as avg_income
            FROM (
                SELECT 
                    strftime('%Y-%m', date) as month,
                    SUM(amount) as monthly_income
                FROM transactions
                WHERE amount > 0
                GROUP BY month
            )
        """)
        
        income_result = cur.fetchone()
        avg_monthly_income = float(income_result['avg_income']) if income_result['avg_income'] else 0
        
        if avg_monthly_income == 0:
            return [{
                'category': 'No Income Data',
                'current_spending': 0,
                'suggested_budget': 0,
                'percentage': 'Upload transactions with income to see budget suggestions',
                'status': 'info',
                'difference': 0
            }]
        
        needs_target = avg_monthly_income * 0.50
        wants_target = avg_monthly_income * 0.20
        savings_target = avg_monthly_income * 0.30
        
        cur.execute("""
            SELECT 
                category,
                AVG(monthly_total) as avg_spending
            FROM (
                SELECT 
                    category,
                    strftime('%Y-%m', date) as month,
                    ABS(SUM(amount)) as monthly_total
                FROM transactions
                WHERE amount < 0
                GROUP BY category, month
            )
            GROUP BY category
        """)
        
        category_spending = {row['category']: float(row['avg_spending']) for row in cur.fetchall()}
        
        needs_actual = sum(category_spending.get(cat, 0) for cat in NEEDS_CATEGORIES)
        wants_actual = sum(category_spending.get(cat, 0) for cat in WANTS_CATEGORIES)
        total_spending = needs_actual + wants_actual
        savings_actual = avg_monthly_income - total_spending
        
        suggestions = []
        
        needs_status = 'on_track' if needs_actual <= needs_target * 1.05 else 'review'
        suggestions.append({
            'category': 'Needs (Groceries, Utilities, Transportation, Health)',
            'current_spending': round(needs_actual, 2),
            'suggested_budget': round(needs_target, 2),
            'percentage': f"{(needs_actual / avg_monthly_income * 100):.1f}% (target: 50%)",
            'status': needs_status,
            'difference': round(needs_target - needs_actual, 2)
        })
        
        wants_status = 'on_track' if wants_actual <= wants_target * 1.05 else 'review'
        suggestions.append({
            'category': 'Wants (Dining, Entertainment, Shopping)',
            'current_spending': round(wants_actual, 2),
            'suggested_budget': round(wants_target, 2),
            'percentage': f"{(wants_actual / avg_monthly_income * 100):.1f}% (target: 20%)",
            'status': wants_status,
            'difference': round(wants_target - wants_actual, 2)
        })
        
        savings_status = 'on_track' if savings_actual >= savings_target * 0.95 else 'review'
        suggestions.append({
            'category': 'Savings & Investments',
            'current_spending': round(savings_actual, 2),
            'suggested_budget': round(savings_target, 2),
            'percentage': f"{(savings_actual / avg_monthly_income * 100):.1f}% (target: 30%)",
            'status': savings_status,
            'difference': round(savings_actual - savings_target, 2)
        })
        
        suggestions.append({
            'category': '--- Detailed Breakdown ---',
            'current_spending': 0,
            'suggested_budget': 0,
            'percentage': '',
            'status': 'info',
            'difference': 0
        })
        
        for category, actual in category_spending.items():
            if category in NEEDS_CATEGORIES:
                category_target = needs_target * (actual / needs_actual) if needs_actual > 0 else 0
                budget_type = "Need"
            elif category in WANTS_CATEGORIES:
                category_target = wants_target * (actual / wants_actual) if wants_actual > 0 else 0
                budget_type = "Want"
            else:
                category_target = actual * 0.9
                budget_type = "Other"
            
            status = 'on_track' if actual <= category_target * 1.1 else 'review'
            
            suggestions.append({
                'category': f"  ↳ {category.title()} ({budget_type})",
                'current_spending': round(actual, 2),
                'suggested_budget': round(category_target, 2),
                'percentage': f"{(actual / avg_monthly_income * 100):.1f}%",
                'status': status,
                'difference': round(category_target - actual, 2)
            })
        
        suggestions.insert(0, {
            'category': 'Average Monthly Income',
            'current_spending': round(avg_monthly_income, 2),
            'suggested_budget': round(avg_monthly_income, 2),
            'percentage': '100%',
            'status': 'info',
            'difference': 0
        })
        
        return suggestions

@app.delete("/transactions")
async def clear_transactions():
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM transactions")
        conn.commit()
    
    categorizer.is_trained = False
    if os.path.exists(categorizer.model_path):
        os.remove(categorizer.model_path)
    
    return {"message": "All transactions cleared and model reset"}

@app.put("/transaction/{transaction_id}/category")
async def update_category(transaction_id: int, category: str):
    """Allow user to correct a category and retrain model."""
    with get_db() as conn:
        cur = conn.cursor()
        
        cur.execute("UPDATE transactions SET category = ? WHERE id = ?", (category, transaction_id))
        conn.commit()
        
        cur.execute("SELECT description, category FROM transactions WHERE category != 'other'")
        all_data = cur.fetchall()
        
        if len(all_data) >= 5:
            descriptions = [row['description'] for row in all_data]
            categories = [row['category'] for row in all_data]
            
            if categorizer.train(descriptions, categories):
                return {"message": "Category updated and model retrained!"}
    
    return {"message": "Category updated"}

@app.get("/model-stats")
async def model_stats():
    """Get information about the ML model."""
    with get_db() as conn:
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) as total FROM transactions")
        total = cur.fetchone()['total']
        
        cur.execute("SELECT COUNT(*) as labeled FROM transactions WHERE category != 'other'")
        labeled = cur.fetchone()['labeled']
    
    return {
        "is_trained": categorizer.is_trained,
        "total_transactions": total,
        "labeled_transactions": labeled,
        "using_ml": categorizer.is_trained,
        "model_file": "exists" if os.path.exists(categorizer.model_path) else "not found"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)