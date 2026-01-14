# =============================================================================
# DIAGNOSTIC CELL: Apple Inc Data Analysis
# =============================================================================
# Run this cell to diagnose the data issues with Apple Inc
# Copy the FULL output and share it back for analysis
# =============================================================================

import pandas as pd
import numpy as np

print("=" * 80)
print("DIAGNOSTIC: APPLE INC DATA ANALYSIS")
print("=" * 80)

# =============================================================================
# STEP 1: Check if EDGAR loader is available and loaded
# =============================================================================
print("\n" + "-" * 60)
print("STEP 1: EDGAR Loader Status")
print("-" * 60)

edgar_loaded = False
try:
    if 'edgar_loader' in dir() or 'edgar_loader' in globals():
        loader = globals().get('edgar_loader') or locals().get('edgar_loader')
        if loader and hasattr(loader, 'is_loaded'):
            edgar_loaded = loader.is_loaded
            print(f"[INFO] EDGAR Loader exists: True")
            print(f"[INFO] EDGAR Loader is_loaded: {edgar_loaded}")
        else:
            print("[INFO] EDGAR Loader exists but not properly initialized")
    else:
        print("[INFO] EDGAR Loader not found in namespace")
        print("[INFO] Will try to load JarFraud data directly instead")
except Exception as e:
    print(f"[ERROR] Checking EDGAR loader: {e}")

# =============================================================================
# STEP 2: Load JarFraud dataset and check Apple-related entries
# =============================================================================
print("\n" + "-" * 60)
print("STEP 2: JarFraud Dataset Analysis")
print("-" * 60)

try:
    csv_url = "https://raw.githubusercontent.com/JarFraud/FraudDetection/master/data_FraudDetection_JAR2020.csv"
    print(f"[INFO] Loading JarFraud dataset...")
    df = pd.read_csv(csv_url)
    print(f"[OK] Loaded {len(df)} records")

    # Show all columns
    print(f"\n[INFO] Available columns ({len(df.columns)} total):")
    print(df.columns.tolist())

    # Show data types and sample values
    print(f"\n[INFO] Key financial columns - data types and ranges:")
    key_cols = ['gvkey', 'fyear', 'sale', 'cogs', 'ni', 'oancf', 'at', 'rect', 'invt', 'ap', 'misstate']
    for col in key_cols:
        if col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            if dtype in ['float64', 'int64']:
                print(f"   {col:12s}: dtype={dtype}, non-null={non_null:,}, min={df[col].min():,.2f}, max={df[col].max():,.2f}, mean={df[col].mean():,.2f}")
            else:
                print(f"   {col:12s}: dtype={dtype}, non-null={non_null:,}")
        else:
            print(f"   {col:12s}: NOT FOUND")

except Exception as e:
    print(f"[ERROR] Loading JarFraud: {e}")
    df = None

# =============================================================================
# STEP 3: Check for specific data issues
# =============================================================================
print("\n" + "-" * 60)
print("STEP 3: Data Quality Checks")
print("-" * 60)

if df is not None:
    # Check for cases where net income > revenue (impossible)
    if 'ni' in df.columns and 'sale' in df.columns:
        impossible_ni = df[(df['ni'] > df['sale']) & (df['sale'] > 0)]
        print(f"\n[CHECK] Cases where Net Income > Revenue (should be 0):")
        print(f"   Count: {len(impossible_ni)}")
        if len(impossible_ni) > 0:
            print(f"   Sample (first 5):")
            print(impossible_ni[['gvkey', 'fyear', 'sale', 'ni']].head())

    # Check for extreme DSO values
    if 'rect' in df.columns and 'sale' in df.columns:
        df['calc_dso'] = (df['rect'] / df['sale']) * 365
        extreme_dso = df[df['calc_dso'] > 300]
        print(f"\n[CHECK] Cases with DSO > 300 days:")
        print(f"   Count: {len(extreme_dso)}")
        if len(extreme_dso) > 0:
            print(f"   Sample (first 5):")
            print(extreme_dso[['gvkey', 'fyear', 'sale', 'rect', 'calc_dso']].head())

    # Check for missing CFO
    if 'oancf' in df.columns:
        missing_cfo = df['oancf'].isna().sum()
        print(f"\n[CHECK] Missing CFO (oancf) values:")
        print(f"   Count: {missing_cfo} ({missing_cfo/len(df)*100:.1f}%)")

# =============================================================================
# STEP 4: Check what data the UI is receiving
# =============================================================================
print("\n" + "-" * 60)
print("STEP 4: Simulating Analysis Input")
print("-" * 60)

# If the user entered data manually, let's check what format it expects
print("""
[INFO] The Gradio UI expects financial data in this format:

financials = {
    'revenue': <float>,           # Annual revenue (in dollars, not millions)
    'cogs': <float>,              # Cost of goods sold
    'net_income': <float>,        # Net income
    'cfo': <float>,               # Cash from operations
    'total_assets': <float>,      # Total assets
    'accounts_receivable': <float>, # AR balance
    'inventory': <float>,         # Inventory balance
}

[QUESTION] How did you input Apple data?
   A) Manually entered values in the UI?
   B) Used EDGAR loader with CIK search?
   C) Other method?
""")

# =============================================================================
# STEP 5: Show expected Apple FY2023 values for comparison
# =============================================================================
print("\n" + "-" * 60)
print("STEP 5: Expected Apple Inc FY2023 Values (for comparison)")
print("-" * 60)

print("""
ACTUAL APPLE INC FY2023 (from 10-K filing):
------------------------------------------
Revenue:              $383.3 billion  (383,285,000,000)
Net Income:           $97.0 billion   (96,995,000,000)
Operating Cash Flow:  $110.5 billion  (110,543,000,000)
Total Assets:         $352.6 billion  (352,583,000,000)
Accounts Receivable:  $29.5 billion   (29,508,000,000)
Inventory:            $6.3 billion    (6,331,000,000)

Calculated Ratios:
- DSO (AR/Revenue*365): 28 days  (normal for Apple)
- DIO (Inv/COGS*365):   9 days   (very efficient)
- Net Margin:           25.3%    (healthy)
- CFO/Revenue:          28.8%    (strong cash generation)
""")

# =============================================================================
# STEP 6: Check for unit scaling issues
# =============================================================================
print("\n" + "-" * 60)
print("STEP 6: Unit Scaling Analysis")
print("-" * 60)

print("""
YOUR OUTPUT SHOWED:
- Revenue: $29.357B
- Net Income: $96.995B

POSSIBLE INTERPRETATIONS:

Scenario A: Revenue is AR (Accounts Receivable)
   - Apple AR FY2023 = $29.5B (MATCHES YOUR $29.357B!)
   - This suggests 'revenue' field is incorrectly mapped to AR

Scenario B: Unit mismatch
   - If Revenue was entered as 29357 (no scaling) vs 383285 (millions)
   - The system might be misinterpreting units

Scenario C: Wrong fiscal year or company
   - Data might be from wrong period

[MOST LIKELY]: Your 'revenue' value ($29.357B) matches Apple's AR, not Revenue.
This suggests a field mapping error in the UI or data extraction.
""")

# =============================================================================
# STEP 7: Test the detect_edge_anomalies function with correct data
# =============================================================================
print("\n" + "-" * 60)
print("STEP 7: Test with Correct Apple Data")
print("-" * 60)

# Correct Apple FY2023 data
apple_correct = {
    'revenue': 383285000000,        # $383.3B
    'cogs': 214137000000,           # $214.1B
    'net_income': 96995000000,      # $97.0B
    'cfo': 110543000000,            # $110.5B
    'total_assets': 352583000000,   # $352.6B
    'accounts_receivable': 29508000000,  # $29.5B
    'inventory': 6331000000,        # $6.3B
    'gross_profit': 169148000000,   # $169.1B
}

print("Correct Apple FY2023 financials:")
for k, v in apple_correct.items():
    print(f"   {k:22s}: ${v/1e9:,.2f}B")

# Calculate ratios
print("\nCalculated ratios with CORRECT data:")
dso = (apple_correct['accounts_receivable'] / apple_correct['revenue']) * 365
dio = (apple_correct['inventory'] / apple_correct['cogs']) * 365
cfo_ratio = apple_correct['cfo'] / apple_correct['revenue']
accrual = (apple_correct['net_income'] - apple_correct['cfo']) / apple_correct['total_assets']

print(f"   DSO: {dso:.1f} days (should be ~28 days)")
print(f"   DIO: {dio:.1f} days (should be ~11 days)")
print(f"   CFO/Revenue: {cfo_ratio:.1%} (should be ~29%)")
print(f"   Accrual Ratio: {accrual:.1%} (should be ~-4%)")

# =============================================================================
# STEP 8: Reproduce your error
# =============================================================================
print("\n" + "-" * 60)
print("STEP 8: Reproducing Your Error")
print("-" * 60)

# Your reported values
your_values = {
    'revenue': 29357000000,         # Your $29.357B (likely AR!)
    'net_income': 96995000000,      # Your $96.995B (correct NI)
    'cfo': 110543000000,            # Assumed CFO
    'accounts_receivable': 29508000000,
    'total_assets': 352583000000,
}

print("YOUR input values (reconstructed):")
print(f"   Revenue: ${your_values['revenue']/1e9:.3f}B")
print(f"   Net Income: ${your_values['net_income']/1e9:.3f}B")

# Calculate what DSO would be with your values
if your_values['revenue'] > 0:
    your_dso = (your_values['accounts_receivable'] / your_values['revenue']) * 365
    print(f"\nWith YOUR 'revenue' value:")
    print(f"   DSO = (AR / Revenue) * 365 = ({your_values['accounts_receivable']/1e9:.1f}B / {your_values['revenue']/1e9:.1f}B) * 365")
    print(f"   DSO = {your_dso:.0f} days  <-- This matches your 367 days!")

print("""
[DIAGNOSIS COMPLETE]
====================
The issue is almost certainly that 'revenue' is being populated with
'accounts_receivable' data instead of actual revenue.

Check your EDGAR_TAG_MAPPING or manual data entry to fix this.
""")

print("\n" + "=" * 80)
print("END OF DIAGNOSTIC")
print("=" * 80)
