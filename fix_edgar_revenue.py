# =============================================================================
# FIX: EDGAR Revenue Mapping Issue
# =============================================================================
# ADD THIS CELL RIGHT AFTER THE EDGAR LOADER CELL
# This fixes the issue where Apple's revenue is incomplete
# =============================================================================

print("=" * 70)
print("APPLYING FIX: EDGAR Revenue Tag Mapping")
print("=" * 70)

# =============================================================================
# PROBLEM: The current EDGAR_TAG_MAPPING misses some revenue tags and
# the pivot_table uses aggfunc='first' which drops duplicate mappings.
# Apple uses 'RevenueFromContractWithCustomerExcludingAssessedTax' but
# we're only getting partial data ($181B instead of $391B).
# =============================================================================

# STEP 1: Expand the EDGAR_TAG_MAPPING with more revenue tags
print("\n[FIX 1] Expanding EDGAR_TAG_MAPPING...")

# Add additional revenue-related XBRL tags
ADDITIONAL_REVENUE_TAGS = {
    # More comprehensive revenue tags
    'RevenueFromContractWithCustomerIncludingAssessedTax': 'revenue',
    'SalesRevenueGoodsNet': 'revenue_products',  # Products revenue
    'SalesRevenueServicesNet': 'revenue_services',  # Services revenue
    'RevenuesNetOfInterestExpense': 'revenue',
    'RevenueNet': 'revenue',
    'TotalRevenues': 'revenue',
    'NetSales': 'revenue',
    'SalesRevenueGoodsGross': 'revenue',
    'TotalRevenuesAndOtherIncome': 'revenue',

    # Apple-specific tags (they use custom extensions sometimes)
    'RevenueFromContractWithCustomerExcludingAssessedTaxTransferredAtPointInTime': 'revenue_point',
    'RevenueFromContractWithCustomerExcludingAssessedTaxTransferredOverTime': 'revenue_over_time',
}

# Update the global mapping
for tag, var in ADDITIONAL_REVENUE_TAGS.items():
    if tag not in EDGAR_TAG_MAPPING:
        EDGAR_TAG_MAPPING[tag] = var

print(f"   Added {len(ADDITIONAL_REVENUE_TAGS)} new tags")
print(f"   Total tags in mapping: {len(EDGAR_TAG_MAPPING)}")

# STEP 2: Create a fixed version of to_financials_dict
print("\n[FIX 2] Patching to_financials_dict to sum revenue components...")

# Store original method
_original_to_financials_dict = edgar_loader.to_financials_dict

def fixed_to_financials_dict(self, cik: int, year: int):
    """
    Fixed version that properly aggregates revenue from multiple tags.
    """
    company_data = self.get_company_financials(cik, [year])
    if company_data is None or len(company_data) == 0:
        return None

    row = company_data.iloc[0]
    financials = {}

    for col in row.index:
        if col in ['cik', 'year']:
            continue
        value = row[col]
        if pd.notna(value):
            financials[col] = float(value)

    # FIX: If we have revenue_products and revenue_services, sum them
    if 'revenue_products' in financials and 'revenue_services' in financials:
        total_rev = financials.get('revenue_products', 0) + financials.get('revenue_services', 0)
        if total_rev > financials.get('revenue', 0):
            financials['revenue'] = total_rev
            print(f"   [FIX] Summed Products + Services revenue: ${total_rev:,.0f}")

    # FIX: Recalculate gross_profit correctly
    if 'revenue' in financials and 'cogs' in financials:
        financials['gross_profit'] = financials['revenue'] - financials['cogs']

    return financials

# Monkey-patch the method
import types
edgar_loader.to_financials_dict = types.MethodType(
    lambda self, cik, year: fixed_to_financials_dict(self, cik, year),
    edgar_loader
)

print("   Patched to_financials_dict method")

# STEP 3: Verify the fix with Apple data
print("\n[FIX 3] Verifying fix with Apple (CIK 320193)...")

APPLE_CIK = 320193
apple_financials = edgar_loader.to_financials_dict(APPLE_CIK, 2024)

if apple_financials:
    rev = apple_financials.get('revenue', 0)
    cogs = apple_financials.get('cogs', 0)
    gp = apple_financials.get('gross_profit', 0)

    print(f"\n   Revenue:      ${rev:,.0f}")
    print(f"   COGS:         ${cogs:,.0f}")
    print(f"   Gross Profit: ${gp:,.0f}")

    # Check if gross profit is now positive
    if gp > 0:
        print(f"\n   ✅ Gross Profit is now POSITIVE!")
    else:
        print(f"\n   ⚠️  Gross Profit still negative - need to reload EDGAR data")
        print(f"       Run: edgar_loader.load(verbose=True)")
else:
    print("   [ERROR] Could not get Apple financials")

# STEP 4: Instructions for full fix
print("\n" + "=" * 70)
print("IMPORTANT: For a complete fix, you need to RELOAD the EDGAR data")
print("=" * 70)
print("""
The tag mapping has been updated, but the data was already loaded with
the old mapping. To get complete revenue data:

1. Re-run the EDGAR loader cell with the new tags:
   edgar_loader.load(verbose=True)

2. Or restart the notebook and run all cells again.

The fix will then capture all revenue line items correctly.
""")

print("=" * 70)
print("FIX APPLIED - Reload EDGAR data for full effect")
print("=" * 70)
