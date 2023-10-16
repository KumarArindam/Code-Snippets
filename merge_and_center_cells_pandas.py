import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment

# Sample DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'B'],
    'Value': [10, 20, 15, 30, 25]
}

df = pd.DataFrame(data)

# Create a new Excel workbook
workbook = Workbook()
worksheet = workbook.active

# Write column names to the first row
for col_idx, column_name in enumerate(df.columns, start=1):
    worksheet.cell(row=1, column=col_idx, value=column_name)

# Convert the Pandas DataFrame data to the Excel sheet
for r_idx, row_data in enumerate(df.values, start=2):
    for c_idx, value in enumerate(row_data, start=1):
        worksheet.cell(row=r_idx, column=c_idx, value=value)

# Define a dictionary to track merged cells
merged_cells = {}

# Iterate through the DataFrame to find and merge cells with the same value in the 'Category' column
for row in range(2, len(df) + 2):
    cell_value = worksheet.cell(row=row, column=1).value
    if cell_value not in merged_cells:
        merged_cells[cell_value] = [row]
    else:
        merged_cells[cell_value].append(row)

# Merge and center the cells
for cell_value, row_indices in merged_cells.items():
    if len(row_indices) > 1:
        start_row = row_indices[0]
        end_row = row_indices[-1]
        worksheet.merge_cells(f'A{start_row}:A{end_row}')
        worksheet.cell(row=start_row, column=1).alignment = Alignment(horizontal='center', vertical='center')

# Save the Excel file
workbook.save('merged_and_centered.xlsx')
