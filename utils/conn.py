import psycopg2
from psycopg2 import sql
from datetime import datetime, timedelta
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

# Example usage
student_name = 'jj'  # Replace this with the actual student name
today_date = datetime.now().date()

# Check if an entry exists for the given name and today's date
cursor.execute("SELECT entrytime, exittime FROM public.attendance_log WHERE username = %s AND date = %s", (student_name, today_date))
existing_entry = cursor.fetchone()

if existing_entry:
    entry_time, exit_time = existing_entry

    if entry_time and not exit_time:
        # Entry time exists, but exit time is null, update exit time
        cursor.execute("UPDATE public.attendance_log SET entrytime = %s WHERE username = %s AND date = %s", (datetime.now(), student_name, today_date))
        print(f"Exit time updated for {student_name} on {today_date}")
    else:
        # Entry time doesn't exist, update entry time
        cursor.execute("UPDATE public.attendance_log SET exittime = %s WHERE username = %s AND date = %s", (datetime.now(), student_name, today_date))
        print(f"Entry time updated for {student_name} on {today_date}")
else:
    # No entry for the given name and today's date, insert a new row
    cursor.execute("INSERT INTO public.attendance_log (username, date, entrytime) VALUES (%s, %s, %s)", (student_name, today_date, datetime.now()))
    print(f"New entry created for {student_name} on {today_date}")

# Commit the changes
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()