
import MetaTrader5 as mt5
import time
import sys
from datetime import datetime

# --- CONFIGURAZIONE HIGH ROI ---
SYMBOL = "XAUUSD"
VOLUME = 0.03             # Size fissa per ogni livello
STEP_POINTS = 250         # Distanza per aggiungere (2.5$)
SL_PROTECTION = 50        # Punti di profitto garantito quando si sposta lo SL
MAX_LAYERS = 4            # Massimo numero di posizioni simultanee
MAGIC_NUMBER = 999        # ID per riconoscere i trade del bot

def main():
    if not mt5.initialize():
        print(f"MetaTrader5 initialization failed, error code: {mt5.last_error()}")
        return

    print(f"🚀 PYRAMIDER ENGINE AVVIATO su {SYMBOL}. In attesa di posizioni...")
    print(f"Config: Vol={VOLUME}, Step={STEP_POINTS}, MaxLayers={MAX_LAYERS}")

    while True:
        try:
            # Ottieni posizioni aperte solo su XAUUSD
            positions = mt5.positions_get(symbol=SYMBOL)
            
            if positions is None:
                print(f"Error getting positions: {mt5.last_error()}")
                time.sleep(5)
                continue

            if len(positions) > 0:
                # Ordina posizioni per data di apertura
                sorted_pos = sorted(positions, key=lambda x: x.time)
                first_pos = sorted_pos[0]   # La "Madre"
                last_pos = sorted_pos[-1]   # L'ultima aperta
                
                direction = first_pos.type  # 0 = BUY, 1 = SELL
                count = len(sorted_pos)
                
                # Get current price
                tick = mt5.symbol_info_tick(SYMBOL)
                if tick is None:
                    continue
                    
                current_price = tick.bid if direction == 0 else tick.ask
                symbol_info = mt5.symbol_info(SYMBOL)
                if symbol_info is None:
                    continue
                    
                point = symbol_info.point

                # --- LOGICA BUY ---
                if direction == 0: 
                    profit_distance = (current_price - last_pos.price_open) / point
                    
                    # 1. Scaling In (Aggiungi posizione)
                    if profit_distance >= STEP_POINTS and count < MAX_LAYERS:
                        print(f"Momentum rilevato (+{profit_distance:.1f} pts). Apro Layer {count + 1}...")
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": SYMBOL,
                            "volume": VOLUME,
                            "type": mt5.ORDER_TYPE_BUY,
                            "price": tick.ask,
                            # SL Iniziale sul prezzo dell'ultima (lo SL safe)
                            "sl": last_pos.price_open, 
                            "magic": MAGIC_NUMBER,
                            "comment": f"Pyramid Layer {count+1}",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(request)
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"Order failed: {result.comment}")
                        else:
                            print(f"Layer {count+1} executed!")
                        time.sleep(1) # Wait for execution

                    # 2. Trailing Stop Collettivo (Breakven Aggressivo)
                    # Se l'ultima posizione è in profitto, sposta TUTTI gli SL al prezzo della penultima
                    # (o comunque per proteggere il profitto)
                    if count > 1 and profit_distance >= (STEP_POINTS / 2):
                        # Calcola nuovo SL: sotto l'attuale prezzo di un po'
                        # Qui usiamo la logica user: SL al prezzo della penultima o Entry precedente
                        # L'user code diceva: last_pos.price_open + SL_PROTECTION
                        # Ma last_pos è l'ultima aperta.
                        new_sl = last_pos.price_open + (SL_PROTECTION * point) 
                        
                        # Verifichiamo se new_sl ha senso (deve essere < current_price)
                        if new_sl < current_price:
                            for pos in positions:
                                # Sposta solo se miglioriamo (alziamo) lo SL
                                if pos.sl < new_sl: 
                                    req_sl = {
                                        "action": mt5.TRADE_ACTION_SLTP,
                                        "position": pos.ticket,
                                        "sl": new_sl,
                                        "tp": pos.tp,
                                        "symbol": SYMBOL
                                    }
                                    mt5.order_send(req_sl)

                # --- LOGICA SELL (Speculare) ---
                elif direction == 1:
                    profit_distance = (last_pos.price_open - current_price) / point
                    
                    if profit_distance >= STEP_POINTS and count < MAX_LAYERS:
                        print(f"Momentum rilevato (Down +{profit_distance:.1f}). Apro Layer {count + 1}...")
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": SYMBOL,
                            "volume": VOLUME,
                            "type": mt5.ORDER_TYPE_SELL,
                            "price": tick.bid,
                            "sl": last_pos.price_open,
                            "magic": MAGIC_NUMBER,
                            "comment": f"Pyramid Layer {count+1}",
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_IOC,
                        }
                        result = mt5.order_send(request)
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print(f"Order failed: {result.comment}")
                        else:
                            print(f"Layer {count+1} executed!")
                        time.sleep(1)

                    if count > 1 and profit_distance >= (STEP_POINTS / 2):
                        new_sl = last_pos.price_open - (SL_PROTECTION * point)
                        if new_sl > current_price:
                            for pos in positions:
                                if pos.sl > new_sl or pos.sl == 0:
                                    req_sl = {
                                        "action": mt5.TRADE_ACTION_SLTP,
                                        "position": pos.ticket,
                                        "sl": new_sl,
                                        "tp": pos.tp,
                                        "symbol": SYMBOL
                                    }
                                    mt5.order_send(req_sl)

            time.sleep(1) # HFT Check Rate

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
