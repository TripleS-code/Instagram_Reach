import pandas as pd
from tkinter import Tk, filedialog

def upload_excel():
    """Function to upload an Excel file and return the DataFrame."""
    # Open file dialog to select Excel file
    Tk().withdraw()  # Prevent Tkinter window from appearing
    file_path = filedialog.askopenfilename()
    if file_path:
        return pd.read_excel(file_path)
    else:
        print("No file selected.")
        return None

def filter_data(df, city=None, road_length=None, cyclists=None, hours=None):
    """Function to filter the DataFrame based on provided criteria."""
    if city:
        df = df[df['City'] == city]
    if road_length:
        df = df[df['RoadLength'] == road_length]
    if cyclists:
        df = df[df['Cyclists'] == cyclists]
    if hours:
        df = df[df['Hours'] == hours]
    return df

def categorize_data(df):
    """Function to categorize cyclists based on hours spent on the road."""
    conditions = [
        (df['Hours'] < 1),
        (df['Hours'] >= 1) & (df['Hours'] <= 2),
        (df['Hours'] > 2)
    ]
    categories = ['Less than 1 hour', '1-2 hours', 'More than 2 hours']
    df['Category'] = pd.cut(df['Hours'], bins=[-float('inf'), 1, 2, float('inf')], labels=categories)
    return df

def save_to_excel(df, output_file='output.xlsx'):
    """Function to save the DataFrame to an Excel file."""
    df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")

def main():
    # Step 1: Upload the Excel file
    df = upload_excel()
    if df is None:
        return

    # Step 2: Display DataFrame to understand the structure
    print("Uploaded Data:")
    print(df.head())

    # Step 3: Filter Data (you can modify these parameters as needed)
    city = input("Enter city to filter by (or press Enter to skip): ")
    road_length = input("Enter road length to filter by (or press Enter to skip): ")
    cyclists = input("Enter number of cyclists to filter by (or press Enter to skip): ")
    hours = input("Enter hours spent on road to filter by (or press Enter to skip): ")

    # Convert inputs to appropriate types
    road_length = float(road_length) if road_length else None
    cyclists = int(cyclists) if cyclists else None
    hours = float(hours) if hours else None

    # Filter the data
    filtered_df = filter_data(df, city=city, road_length=road_length, cyclists=cyclists, hours=hours)

    # Step 4: Categorize Data
    categorized_df = categorize_data(filtered_df)

    # Step 5: Display the categorized DataFrame
    print("Categorized Data:")
    print(categorized_df)

    # Step 6: Save the DataFrame to a new Excel file
    save_to_excel(categorized_df)

if __name__ == "__main__":
    main()
