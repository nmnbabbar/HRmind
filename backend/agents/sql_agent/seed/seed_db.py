import asyncio
import aiosqlite
from pathlib import Path
from backend.config import get_settings

settings = get_settings()

CREATE_TABLES = """
DROP TABLE IF EXISTS leave_balances;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    department TEXT NOT NULL,
    role TEXT NOT NULL,
    salary INTEGER NOT NULL,
    hire_date DATE NOT NULL
);

CREATE TABLE leave_balances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    annual_leave_days INTEGER NOT NULL,
    sick_leave_days INTEGER NOT NULL,
    FOREIGN KEY(employee_id) REFERENCES employees(id)
);
"""

EMPLOYEES = [
    ("John", "Doe", "john.doe@company.com", "Engineering", "Software Engineer", 120000, "2021-03-15"),
    ("Jane", "Smith", "jane.smith@company.com", "Engineering", "Senior Software Engineer", 150000, "2019-07-22"),
    ("Alice", "Johnson", "alice.johnson@company.com", "Sales", "Account Executive", 95000, "2022-01-10"),
    ("Bob", "Williams", "bob.williams@company.com", "Sales", "Sales Manager", 130000, "2018-11-05"),
    ("Charlie", "Brown", "charlie.brown@company.com", "HR", "HR Generalist", 75000, "2023-04-18"),
    ("Diana", "Prince", "diana.prince@company.com", "Engineering", "Engineering Manager", 180000, "2017-09-30"),
    ("Evan", "Wright", "evan.wright@company.com", "Marketing", "Marketing Specialist", 85000, "2021-08-12"),
    ("Fiona", "Clark", "fiona.clark@company.com", "Engineering", "Frontend Developer", 110000, "2022-05-23"),
    ("George", "Miller", "george.miller@company.com", "Finance", "Financial Analyst", 90000, "2020-02-14"),
    ("Hannah", "Davis", "hannah.davis@company.com", "Finance", "CFO", 250000, "2015-06-01"),
]

# Generate more mock data up to ~50
for i in range(11, 51):
    dept = "Engineering" if i % 3 == 0 else ("Sales" if i % 3 == 1 else "Support")
    EMPLOYEES.append((
        f"First{i}", f"Last{i}", f"employee{i}@company.com", dept, "Staff", 60000 + (i * 1000), f"202{i%4}-01-01"
    ))

async def seed():
    db_path = Path(settings.sqlite_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiosqlite.connect(db_path) as db:
        await db.executescript(CREATE_TABLES)
        
        # Clear existing
        await db.execute("DELETE FROM leave_balances")
        await db.execute("DELETE FROM employees")
        await db.execute("DELETE FROM users")
        
        # Insert employees
        cursor = await db.cursor()
        for emp in EMPLOYEES:
            await cursor.execute(
                "INSERT INTO employees (first_name, last_name, email, department, role, salary, hire_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                emp
            )
            emp_id = cursor.lastrowid
            
            # Insert leave balances
            annual_leave = 20 if emp[4] != "CFO" else 30
            sick_leave = 10
            await db.execute(
                "INSERT INTO leave_balances (employee_id, annual_leave_days, sick_leave_days) VALUES (?, ?, ?)",
                (emp_id, annual_leave, sick_leave)
            )
            
        await db.commit()
        print(f"Successfully seeded {len(EMPLOYEES)} employees and their leave balances into {db_path}.")

if __name__ == "__main__":
    asyncio.run(seed())
