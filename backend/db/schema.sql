-- backend/db/schema.sql
-- HR Database Schema for SQL Agent queries

-- Enable foreign keys
PRAGMA foreign_keys = ON;

-- Drop tables if they exist to allow clean seeding
DROP TABLE IF EXISTS salary_history;
DROP TABLE IF EXISTS leave_balances;
DROP TABLE IF EXISTS employees;
DROP TABLE IF EXISTS departments;

CREATE TABLE departments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    manager_id INTEGER -- References employees.id, but left loose to avoid circular dependency on creation
);

CREATE TABLE employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    department_id INTEGER,
    job_title TEXT NOT NULL,
    hire_date DATE NOT NULL,
    is_active BOOLEAN DEFAULT 1,
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

CREATE TABLE leave_balances (
    employee_id INTEGER PRIMARY KEY,
    annual_leave_days INTEGER DEFAULT 25,
    sick_leave_days INTEGER DEFAULT 10,
    maternity_paternity_leave_days INTEGER DEFAULT 0,
    FOREIGN KEY (employee_id) REFERENCES employees(id)
);

CREATE TABLE salary_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL,
    salary_amount DECIMAL(10,2) NOT NULL,
    effective_date DATE NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(id)
);
