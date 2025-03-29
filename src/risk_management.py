def suggest_stop_loss(current_price, purchase_price, stop_loss_percent=5):
    stop_loss_price = purchase_price * (1 - stop_loss_percent / 100)
    if current_price <= stop_loss_price:
        print(f"ALERT 🚨: Stop-loss triggered for {current_price}. Consider selling.")
    else:
        print(f"INFO: {current_price} is above the stop-loss price of {stop_loss_price}. Holding position.")

def suggest_hedging(asset, exposure, hedge_percent=10):
    hedge_amount = exposure * hedge_percent / 100
    print(f"Consider hedging {hedge_amount} worth of {asset} to mitigate risk.")