# Portfolio Strategy Comparison

This project analyzes different investment strategies starting from **$1,000 on April 7, 2025**, using data from Refinitiv Workspace.  
The strategies include:

- **Cash_0** — $1,000 in a savings account at 0%.
- **HYSA_3p5** — $1,000 in a 3.5% APY high-yield savings account.
- **BH_Combined** — Buy & Hold with equal allocation across ETFs (QQQ/QQQM, SPY, VOO, VTI, SCHD).
- **Signal_EQW_MA** — A dynamic signal strategy: equal-weighting only assets with **MA(50) > MA(200)**; idle cash earns HYSA.

---

## 📊 Results
- **Buy & Hold (BH_Combined)** outperformed with ~25% return since start.
- **Signal_EQW_MA** achieved ~14% return, lower but with smaller drawdowns.
- **HYSA_3p5** provided stability with ~1.3% growth.
- **Cash_0** remained flat at $1,000.

Sharpe Ratios (excess over HYSA):
- **BH_Combined:** 2.74  
- **Signal_EQW_MA:** 1.77  
- **HYSA_3p5:** 0.00  
- **Cash_0:** 0.00  

---

## 🧾 Interpretation
- **Conclusion:** Buy & Hold has been the most profitable and less risky strategy since April 2025.  
- **Recommendation:** Long-term investors should favor diversified Buy & Hold allocations.  
- **Limitations:** Short test period, no transaction costs, limited assets.  
- **Future Plans:** Extend to multiple years, include credit derivatives/options, stress-test with volatility shocks, and automate reporting.

---

## 🚀 Usage
1. Place your Refinitiv API key in `api_key.txt` (gitignored).
2. Run the Jupyter Notebook.
3. Visuals and tables will be generated comparing portfolio strategies.

---
