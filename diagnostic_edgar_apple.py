# =============================================================================
# DIAGNOSTIC: Deep EDGAR Data Analysis for Apple
# =============================================================================
# This checks what XBRL tags Apple actually uses and how they map
# =============================================================================

print("=" * 80)
print("DEEP DIAGNOSTIC: Apple EDGAR Data")
print("=" * 80)

# Apple's CIK
APPLE_CIK = 320193

# =============================================================================
# STEP 1: Get Apple's raw data from EDGAR loader
# =============================================================================
print("\n" + "-" * 60)
print("STEP 1: Apple Raw Data from EDGAR Loader")
print("-" * 60)

try:
    if 'edgar_loader' in globals() and edgar_loader.is_loaded:
        # Get all Apple financials
        apple_data = edgar_loader.get_company_financials(APPLE_CIK)

        if apple_data is not None:
            print(f"[OK] Found Apple data: {len(apple_data)} years")
            print(f"\nYears available: {sorted(apple_data['year'].unique())}")

            print("\n[INFO] All columns in Apple data:")
            for col in sorted(apple_data.columns):
                if col not in ['cik', 'year']:
                    # Get latest value
                    latest = apple_data[apple_data['year'] == apple_data['year'].max()]
                    val = latest[col].iloc[0] if len(latest) > 0 else None
                    if pd.notna(val):
                        print(f"   {col:30s}: ${val:,.0f}")
                    else:
                        print(f"   {col:30s}: NULL")

            # Specifically check revenue vs accounts_receivable
            print("\n" + "-" * 40)
            print("KEY COMPARISON (FY2023 or latest):")
            print("-" * 40)
            latest = apple_data[apple_data['year'] == apple_data['year'].max()].iloc[0]

            rev = latest.get('revenue', None)
            ar = latest.get('accounts_receivable', None)
            ni = latest.get('net_income', None)
            cfo = latest.get('cfo', None)

            print(f"revenue:             {('$' + f'{rev:,.0f}') if pd.notna(rev) else 'NULL'}")
            print(f"accounts_receivable: {('$' + f'{ar:,.0f}') if pd.notna(ar) else 'NULL'}")
            print(f"net_income:          {('$' + f'{ni:,.0f}') if pd.notna(ni) else 'NULL'}")
            print(f"cfo:                 {('$' + f'{cfo:,.0f}') if pd.notna(cfo) else 'NULL'}")

            if pd.notna(rev) and pd.notna(ar):
                print(f"\nIs revenue == AR? {abs(rev - ar) < 1e6}")
                print(f"Revenue / AR ratio: {rev/ar:.2f}")
        else:
            print("[ERROR] No Apple data found")
    else:
        print("[ERROR] EDGAR loader not available")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# STEP 2: Check the raw NUM files for Apple's XBRL tags
# =============================================================================
print("\n" + "-" * 60)
print("STEP 2: Check Raw XBRL Tags for Apple")
print("-" * 60)

try:
    if 'edgar_loader' in globals() and hasattr(edgar_loader, '_adsh_to_cik'):
        # Find Apple's ADSH (filing identifier)
        apple_adsh = edgar_loader._adsh_to_cik[
            edgar_loader._adsh_to_cik['cik'] == APPLE_CIK
        ]

        if len(apple_adsh) > 0:
            print(f"[OK] Found {len(apple_adsh)} Apple filings")
            print("\nApple filing identifiers (ADSH):")
            for _, row in apple_adsh.iterrows():
                print(f"   Year {row['fy']}: {row['adsh']}")
        else:
            print("[WARN] No Apple ADSHs found")
except Exception as e:
    print(f"[ERROR] {e}")

# =============================================================================
# STEP 3: Check the EDGAR_TAG_MAPPING
# =============================================================================
print("\n" + "-" * 60)
print("STEP 3: EDGAR_TAG_MAPPING Check")
print("-" * 60)

print("\nTags mapped to 'revenue':")
for tag, var in EDGAR_TAG_MAPPING.items():
    if var == 'revenue':
        print(f"   {tag}")

print("\nTags mapped to 'accounts_receivable':")
for tag, var in EDGAR_TAG_MAPPING.items():
    if var == 'accounts_receivable':
        print(f"   {tag}")

# =============================================================================
# STEP 4: Test the to_financials_dict output
# =============================================================================
print("\n" + "-" * 60)
print("STEP 4: Test to_financials_dict() Output")
print("-" * 60)

try:
    if 'edgar_loader' in globals() and edgar_loader.is_loaded:
        financials = edgar_loader.to_financials_dict(APPLE_CIK, 2023)

        if financials:
            print(f"\n[OK] to_financials_dict returned {len(financials)} fields:")
            for k, v in sorted(financials.items()):
                print(f"   {k:25s}: ${v:,.0f}")

            # Check the problematic values
            print("\n" + "-" * 40)
            print("CRITICAL CHECK:")
            print("-" * 40)
            rev = financials.get('revenue', 0)
            ar = financials.get('accounts_receivable', 0)

            print(f"revenue value:             ${rev:,.0f}")
            print(f"accounts_receivable value: ${ar:,.0f}")

            # Apple's actual FY2023 revenue is $383B
            APPLE_ACTUAL_REVENUE = 383285000000
            APPLE_ACTUAL_AR = 29508000000

            print(f"\nExpected Apple Revenue:    ${APPLE_ACTUAL_REVENUE:,.0f}")
            print(f"Expected Apple AR:         ${APPLE_ACTUAL_AR:,.0f}")

            if abs(rev - APPLE_ACTUAL_AR) < 1e9:
                print("\n⚠️  WARNING: 'revenue' matches expected AR value!")
                print("   This confirms the field mapping bug.")
            elif abs(rev - APPLE_ACTUAL_REVENUE) < 1e9:
                print("\n✅ GOOD: 'revenue' matches expected Revenue value")
            else:
                print(f"\n❓ UNEXPECTED: revenue doesn't match either expected value")
                print(f"   Difference from expected revenue: ${abs(rev - APPLE_ACTUAL_REVENUE):,.0f}")
                print(f"   Difference from expected AR: ${abs(rev - APPLE_ACTUAL_AR):,.0f}")
        else:
            print("[ERROR] to_financials_dict returned None")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

# =============================================================================
# STEP 5: Alternative - check if data comes from Gradio UI input
# =============================================================================
print("\n" + "-" * 60)
print("STEP 5: Check Gradio UI Data Source")
print("-" * 60)

print("""
If you entered Apple data manually in the Gradio UI, check:

1. Did you copy/paste from a specific source?
2. Which fields did you fill in?
3. Were the values in dollars or millions/billions?

The UI expects values in FULL DOLLARS, not millions.
   - Correct: 383285000000 (383.3 billion)
   - Wrong:   383285 (interpreted as $383K)
""")

print("\n" + "=" * 80)
print("END DIAGNOSTIC - Share this output!")
print("=" * 80)
