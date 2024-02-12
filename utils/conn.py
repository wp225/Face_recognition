import psycopg2
from psycopg2 import sql

# Replace these values with your database credentials
db_host = "110.44.123.230"
db_port = "5432"
db_name = "testdb"
db_user = "test"
db_password = "test@1234"


conn = psycopg2.connect(
    host=db_host,
    port=db_port,
    database=db_name,
    user=db_user,
    password=db_password
)

# Assume 'attendance_log' is the table you want to access
cursor = conn.cursor()
# Example 1: Fetch all rows from the table
cursor.execute("SELECT * FROM public.attendance_log")
rows = cursor.fetchall()
print(rows)
