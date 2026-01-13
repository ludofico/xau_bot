
#property copyright "Antigravity AI"
#property link      "https://github.com/google/antigravity"
#property version   "3.40"
#property strict

input string   InpUrl          = "http://192.168.10.168:5555/poll";
input int      InpPollInterval = 5000;

int tick_count = 0;

int OnInit() {
   Print("=== BridgeClient V3.4 DEBUG ===");
   Print("URL: ", InpUrl);
   Print("Interval: ", InpPollInterval, "ms");
   EventSetMillisecondTimer(InpPollInterval);
   Print("Timer started!");
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
   EventKillTimer();
}

void OnTimer() {
   tick_count++;
   Print("--- Timer tick #", tick_count, " ---");
   
   MqlTick tick;
   if(!SymbolInfoTick(_Symbol, tick)) {
      Print("ERROR: SymbolInfoTick failed for ", _Symbol);
      return;
   }
   
   Print("Tick OK: Bid=", tick.bid, " Ask=", tick.ask);
   
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   Print("Account: Balance=", balance, " Equity=", equity);
   
   string msg = "POLL|" + _Symbol + "|" + 
                DoubleToString(tick.bid, _Digits) + "|" + 
                DoubleToString(tick.ask, _Digits) + "|" + 
                DoubleToString(balance, 2) + "|" + 
                DoubleToString(equity, 2);
   
   Print("Sending: ", msg);
   Print("Calling WebRequest to: ", InpUrl);
   
   char post_data[];
   char result[];
   string result_headers;
   
   StringToCharArray(msg, post_data);
   
   ResetLastError();
   int res = WebRequest("POST", InpUrl, NULL, 3000, post_data, result, result_headers);
   int err = GetLastError();
   
   Print("WebRequest returned: ", res, " | LastError: ", err);
   
   if(res == -1) {
      if(err == 4014) {
         Print("!!! ADD URL TO WHITELIST: Tools->Options->Expert Advisors !!!");
      } else if(err == 5203) {
         Print("!!! NETWORK ERROR - Wine cannot reach IP !!!");
      } else {
         Print("WebRequest failed with error: ", err);
      }
   } else if(res == 200) {
      string response = CharArrayToString(result);
      Print("Response: ", response);
   } else {
      Print("HTTP status: ", res);
   }
}
