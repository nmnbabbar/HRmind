"""
backend/db/seed_db.py
======================
Initializes the HR SQLite database and populates it with realistic mock data.
Used for development and for the SQL Agent to query.
"""

import sqlite3
import random
import datetime
from pathlib import Path

from backend.config import get_settings

def generate_mock_data(db_path: Path, schema_path: Path):
    print(f"Connecting to database at: {db_path}")
    
    # Ensure directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 1. Execute schema
    print(f"Executing schema from: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        cursor.executescript(f.read())
        
    # 2. Insert Departments
    departments = ["Engineering", "Human Resources", "Sales", "Marketing", "Finance"]
    for dept in departments:
        cursor.execute("INSERT INTO departments (name) VALUES (?)", (dept,))
    
    # 3. Insert Employees
    first_names = ["Aarav", "Vihaan", "Aditya", "Sai", "Arjun", "Ananya", "Diya", "Aadhya", "Priya", "Neha", "Rohan", "Rahul", "Pooja", "Sneha", "Karan", "Kavya", "Vikram", "Ishita"]
    last_names = ["Sharma", "Patel", "Singh", "Kumar", "Gupta", "Deshmukh", "Joshi", "Iyer", "Nair", "Reddy", "Verma", "Rao", "Das", "Chauhan"]
    job_titles = ["Software Engineer", "HR Specialist", "Sales Representative", "Marketing Manager", "Financial Analyst", "Data Scientist", "Product Manager"]
    
    employees_data = []
    for i in range(1, 51):
        first = random.choice(first_names)
        last = random.choice(last_names)
        email = f"{first.lower()}.{last.lower()}{i}@hrmind.local"
        dept_id = random.randint(1, len(departments))
        title = random.choice(job_titles)
        
        # Random hire date between 2018 and 2023
        hire_date = datetime.date(
            random.randint(2018, 2023), 
            random.randint(1, 12), 
            random.randint(1, 28)
        )
        is_active = 1 if random.random() > 0.1 else 0  # 90% active
        
        employees_data.append((first, last, email, dept_id, title, hire_date.isoformat(), is_active))
        
    cursor.executemany("""
        INSERT INTO employees (first_name, last_name, email, department_id, job_title, hire_date, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, employees_data)
    
    # 4. Insert Leave Balances & Salary History
    cursor.execute("SELECT id, hire_date FROM employees")
    all_emps = cursor.fetchall()
    
    leave_data = []
    salary_data = []
    
    for emp_id, h_date in all_emps:
        # Leave balances
        annual = random.randint(5, 25)
        sick = random.randint(0, 10)
        leave_data.append((emp_id, annual, sick, 0))
        
        # Salary History (initial + maybe promotion)
        base_salary = random.randint(50000, 120000)
        salary_data.append((emp_id, base_salary, h_date))
        
        # If hired before 2021, give them a raise in 2022
        hire_year = int(h_date[:4])
        if hire_year < 2021:
            raise_date = f"2022-01-01"
            new_salary = base_salary * 1.10
            salary_data.append((emp_id, round(new_salary, 2), raise_date))
            
    cursor.executemany("""
        INSERT INTO leave_balances (employee_id, annual_leave_days, sick_leave_days, maternity_paternity_leave_days)
        VALUES (?, ?, ?, ?)
    """, leave_data)
    
    cursor.executemany("""
        INSERT INTO salary_history (employee_id, salary_amount, effective_date)
        VALUES (?, ?, ?)
    """, salary_data)
    
    # Assign managers to departments from within the department
    for i in range(1, len(departments) + 1):
        cursor.execute("SELECT id FROM employees WHERE department_id = ?", (i,))
        dept_employees = cursor.fetchall()
        if dept_employees:
            manager_id = random.choice(dept_employees)[0]
            cursor.execute("UPDATE departments SET manager_id = ? WHERE id = ?", (manager_id, i))
            # Also update their job title to make it clear they are a manager
            cursor.execute("UPDATE employees SET job_title = 'Department Manager' WHERE id = ?", (manager_id,))
        
    conn.commit()
    conn.close()
    print("Database seeding complete. Inserted 50 employees and related records.")

if __name__ == "__main__":
    settings = get_settings()
    
    # The config sets sqlite_db_path. Let's resolve it against the project root.
    project_root = Path(__file__).parent.parent.parent
    db_file = project_root / "data" / "hr_database.sqlite"
    schema_file = Path(__file__).parent / "schema.sql"
    
    generate_mock_data(db_file, schema_file)
