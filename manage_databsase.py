import sqlite3


def get_user_by_vehicle(vehicle_number):
    print("vehicle_number = ",vehicle_number)
    conn = sqlite3.connect("vehiclesdata.db")
    cursor = conn.cursor()
    cursor.execute("SELECT fullname, email FROM users_info WHERE vehicle_number = ?", (vehicle_number,))
    result = cursor.fetchone()
    if result:
        print(f"Full Name: {result[0]}, Email: {result[1]}")
    else:
        print("No user found with this vehicle number.")
    conn.close()
    return result[0],result[1]

def get_voilation_of_vehicle(vehicle_number):
    conn = sqlite3.connect("vehiclesdata.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM voilation_info WHERE vehicle_number = ?", (vehicle_number,))
    result = cursor.fetchone()
    if result:
      
        print(result)
    else:
        print("No user found with this vehicle number.")
    conn.close()
    return result

def insert_voilation(vehicle_number,voilation):
    conn = sqlite3.connect("vehiclesdata.db")
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO voilation_info (vehicle_number,voilation)
            VALUES ( ?,?)
        ''', (vehicle_number,voilation))
        conn.commit()
        print("User added successfully.")
    except sqlite3.IntegrityError:
        print("Error: Email or Vehicle Number already exists.")
    conn.close()


