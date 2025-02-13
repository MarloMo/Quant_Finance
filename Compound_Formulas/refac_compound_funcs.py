import numpy as np
import matplotlib.pyplot as plt
import compound_funcs as cf

##################################
##### Adjustable Variables ######
##################################
P = 5000  # Principal amount
r = 0.12  # APY (Annual Percentage Yield)
age = 22  # Current Age
retirement_age = 65  # Potential Retirement Age
yearly_contributions = 580 * 12  # Annual Contributions
retirement_output_rate = 0.05  # Rate of Annual Withdrawal During Retirement Years
##################################
##### Adjustable Constraints ######
annual_limit = 7000  # IRA Annual Contribution Limit (2025)
##################################
##################################
##################################
##################################

# Time settings
n = 1  # Compounding frequency (1 for yearly)
target_time = retirement_age - age  # Time until retirement
time = np.linspace(0, target_time, 100)  # Time for compound function
age_spectrum = np.linspace(age, retirement_age, 100)  # Age range

# Calculate the annual and monthly payouts
retirement_balance = cf.ira_compound(
    P, r, target_time, yearly_contributions, annual_limit
)
annual_payout = retirement_balance * retirement_output_rate
monthly_payout = annual_payout / 12

### Plotting Params ###
plt.figure(figsize=(6.5, 6))
plt.style.use("ggplot")

# Plot the estimated IRA balance over time
plt.plot(
    age_spectrum,
    cf.ira_compound(P, r, time, yearly_contributions, annual_limit),
    color="r",
    label=f"APY: {r * 100}% (avg)",
)

# Customize plot appearance
plt.legend()
plt.title(f"Estimated IRA Balance: ~ ${retirement_balance:,.2f}")
plt.xlabel("Age (yr)")
plt.ylabel("Total Savings ($)")

# Add text for additional details
plt.text(
    0.5,  # x-coordinate (fraction of x-axis range)
    -0.30,  # y-coordinate (fraction of y-axis range)
    f"Annual Contributions: ${yearly_contributions:,.2f}/yr \n"
    f"Total years lapsed: {target_time} yrs \n"
    f"Annual Payout: ~ ${annual_payout:,.2f} \n"
    f"Monthly Payout: ~ ${monthly_payout:,.2f}",
    horizontalalignment="center",
    verticalalignment="center",
    transform=plt.gca().transAxes,
)

# Adjust layout and save the plot
plt.tight_layout()
plt.savefig("Refac_IRA_Savings_Chart.pdf")

ira_comp_disc = cf.ira_compound_discrete(
    P, r, time, yearly_contributions, annual_limit
)[-1]
print(
    f"test output: ~ ${ira_comp_disc:.2f}",
)
