// Relays sensor readings from arduino to an IoT server
// by: Emmanuel Agossou
 
// ======= flush the serial before sending data to remove old data ==============
#include<SoftwareSerial.h>
SoftwareSerial mySerial1(0,4 ); //Rx,Tx pins // Rx, TX
#include <WiFi.h>
#include <WiFiClient.h>
#include <PubSubClient.h>
#include <WebServer.h>
#include <Wire.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <HTTPClient.h>


const char* ssid     = "KIC_LOCAL_G";
const char* password = "KICKICKICKICKIC";
String controlString; // Captures out URI querystring;

//const char* ssid = "Buffalo-G-11D0";
//const char* password = "xh6h8bsbhern6";

//const char* ssid     = "manu"; // Your ssid
//const char* password = "living2020"; // Your Password

String apiKey = "KQ25R81ZCGDLUEU3"; // api for Thingspeak write
const char* serverThinkspeak = "api.thingspeak.com";
String output,temperatureDec, pHDec,O2Dec,NH3Dec,TDSDec,ECDec,WaterLevelDec,turbidityDec,CO2Dec; // outpout on webpage of the different decision based on different sensors measurements 
float  temperature, pH,O2,NH3,TDS,EC,WaterLevel,CO2,turbidity;
int waterPumpState=0,oxygenPumpState=0,heatPumpState=0,colderPumpState=0,feederpumpState=0;
int water_PumpPin=32,O2_PumpPin=33,HEAT_EAU_PumpPin=14,COLD_EAU_PumpPin=12,FEED_PumpPin=34;   // pin ESP32 pour chaque pompes

WiFiServer server(80);
//WebServer server(80);

// for MQTT
#define mqtt_port 1883
#define mqtt_server "broker.hivemq.com"
#define MQTT_USER "eFishFarming"
#define MQTT_PASSWORD "eFishFarming"
#define MQTT_SERIAL_PUBLISH_CH "/ic/esp32/serialdata/uno/"

//****** For the online webplatform *********************************************
char status;
// Define NTP Client to get time
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP);
 
// Variables to save date and time
String formattedDate;
String dayStamp;
String timeStamp;
String dateTime;
// REPLACE with your Domain name and URL path or IP address with path
const char* serverName = "http://efishfarm.atwebpages.com/post-data.php";

// Keep this API Key value to be compatible with the PHP code provided in the project page. 
// If you change the apiKeyValue value, the PHP file /post-esp-data.php also needs to have the same key 
String apiKeyValue = "tPmAT5Ac3k7E9";  //The apiKeyValue is just a random string that you can modify. It’s used for security reasons, so only anyone that knows your API key can publish data to your database
String UserID = "000001"; // to be changed in parameter
String SystemID = "1100001"; // to be changed in parameter
String location = "Benin, Calavi";

 String waterPumpStateStr,oxygenPumpStateStr,heatPumpStateStr,colderPumpStateStr,feederpumpStateStr;
//**********************************************************

WiFiClient client;
//WiFiClient MQTTclient;
//PubSubClient client(MQTTclient);

/******** end FOR LCD Display*********/
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
#define OLED_MOSI     9  // D1
#define OLED_CLK      10  // D0
#define OLED_DC       11  
#define OLED_CS       12
#define OLED_RESET     13 // Reset pin # (or -1 if sharing Arduino reset pin)
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, OLED_MOSI,OLED_CLK,OLED_DC,OLED_RESET, OLED_CS);
/******** end FOR LCD Display*********/


// function setup starts
void setup() {
      Serial.begin(115200);
      Serial1.begin(9600);

      
  // OLED Display-initialize with the I2C addr 0x3C
        if(!display.begin(SSD1306_SWITCHCAPVCC)){
            Serial.println("SSD1306 allocation failed");
            for(;;);//Don't proceed, loop forever
        }     
      
      delay(100);
     
      // We start by connecting to a WiFi network
       WiFi.begin(ssid,password);
      Serial.begin(115200);
      while(WiFi.status()!=WL_CONNECTED)
      {
        Serial.print(".");
        delay(500);
      }
      Serial.println("");
      Serial.print("IP Address: ");
      Serial.println(WiFi.localIP());

      mySerial1.begin(9600);

     // Initialize a NTPClient to get time
          timeClient.begin();
          // Set offset time in seconds to adjust for your timezone, for example:
          // GMT +1 = 3600
          // GMT +8 = 28800
          // GMT -1 = -3600
          // GMT 0 = 0
          timeClient.setTimeOffset(32400);

        // pumps state initialization
            pinMode(HEAT_EAU_PumpPin,OUTPUT);
            digitalWrite(HEAT_EAU_PumpPin, LOW);
            heatPumpState=0;
            
}
 
void loop() {  
    remoteControle(); // remote controle function
  // Les pompes
    if(heatPumpState==1){
      digitalWrite(HEAT_EAU_PumpPin, HIGH); // set pin high
      Serial.println("HEAT_EAU_Pump Status: ON"); 
    }
    if(heatPumpState==0){
      digitalWrite(HEAT_EAU_PumpPin, LOW); // set pin high
      Serial.println("HEAT_EAU_Pump Status: OFF"); 
    }

    if(oxygenPumpState==1){
      digitalWrite(O2_PumpPin, HIGH); // set pin high
      Serial.println("O2_Pump Status: ON"); 
    }
    if(oxygenPumpState==0){
      digitalWrite(O2_PumpPin, LOW); // set pin high
      Serial.println("O2_Pump Status: OFF"); 
    }

    if(colderPumpState==1){
      digitalWrite(COLD_EAU_PumpPin, HIGH); // set pin high
      Serial.println("COLD_EAU_Pump Status: ON"); 
    }
    if(colderPumpState==0){
      digitalWrite(COLD_EAU_PumpPin, LOW); // set pin high
      Serial.println("COLD_EAU_Pump Status: OFF"); 
    }

    if(waterPumpState==1){
      digitalWrite(water_PumpPin, HIGH); // set pin high
      Serial.println("water_Pump Status: ON"); 
    }
    if(waterPumpState==0){
      digitalWrite(water_PumpPin, LOW); // set pin high
      Serial.println("water_Pump Status: OFF"); 
    }

    if(feederpumpState==1){
      digitalWrite(FEED_PumpPin, HIGH); // set pin high
      Serial.println("FEED_Pump Status: ON"); 
    }
    if(feederpumpState==0){
      digitalWrite(FEED_PumpPin, LOW); // set pin high
      Serial.println("FEED_Pump Status: OFF"); 
    }
    
  String msg= mySerial1.readStringUntil('\r'); // read from serial monitor data sent by arduino UNO
   Serial.print("Message:");
  Serial.print(msg);
  //split msg
  const char *str1 = msg.c_str();
  char newString[10][10]; 
    int i,j,ctr;
    j=0; ctr=0;
    for(i=0;i<=(strlen(str1));i++)
    {
        // if space or NULL found, assign NULL into newString[ctr]
        if(str1[i]==' '||str1[i]=='\0')
        {
            newString[ctr][j]='\0';
            ctr++;  //for next word
            j=0;    //for next word, init index to 0
        }
        else
        {
            newString[ctr][j]=str1[i];
            j++;
        }
    }

  temperature=atof(newString[0]);
 
  O2=atof(newString[1]);
  turbidity=atof(newString[2]);
 
  CO2=atof(newString[4]);
  WaterLevel=atof(newString[5]);
  NH3=atof(newString[6]);
  TDS=atof(newString[7]);
  // EC=atof(newString[3]);
  EC=TDS*2;
  pH=atof(newString[8]);

 
  float thesum;
  thesum = temperature+O2+turbidity+EC+CO2+WaterLevel+NH3+TDS+pH;
 
  //Prepare the data for serving it over HTTP
  /*
   output="temperature:"+String(temperature)+"\n";
  output+="temperatureC:"+String(temperatureC);
  output+="fanState:"+String(fanState);
  output+="lightState:"+String(lightState);
  output+="eggRotatorState:"+String(eggRotatorState);
  output+="windowsState:"+String(windowsState);
  output+="humidifierState:"+String(humidifierState);
  output+="flameOdor:"+String(flameOdor);
  //serve the data as plain text, for example
  //server.send(200,"text/plain",output);

*/
   int compte=0;
    if(temperature>0) compte++;
    if(O2>0) compte++;
    if(turbidity>0) compte++;
    if(EC>0) compte++;
    if(CO2>0) compte++;
    if(WaterLevel>0) compte++;
    if(NH3>0) compte++;
    if(TDS>0) compte++;
    if(pH>0) compte++;
    
            //
   if(compte>3){
      /*********************************/

      // decisions scripts here
    
        if((temperature<21)&&(temperature>33)) temperatureDec="PAS BON";
          else temperatureDec="BON";

        if((pH<6)&&(pH>9)) pHDec="PAS BON";
          else pHDec="BON";

        if(O2<0) O2Dec="PAS BON";
          else O2Dec="BON";

       if(NH3<0.5) NH3Dec="PAS BON";
          else NH3Dec="BON";

       if((turbidity<200)&&(turbidity>400)) turbidityDec="PAS BON";
          else turbidityDec="BON";

       if(EC<0) ECDec="PAS BON";
          else ECDec="BON";

      if(CO2<0)CO2Dec="PAS BON";
          else CO2Dec="BON";

      if(TDS<0) TDSDec="PAS BON";
          else TDSDec="BON";

      if(WaterLevel<50) WaterLevelDec="PAS BON";
          else WaterLevelDec="BON";
     
          // put your main code here, to run repeatedly:
   // display on LCD     
      /*
      display.setTextSize(1); // Normal 1:1 pixel
      display.setTextColor(WHITE); // Draw white text
       display.clearDisplay(); // clear display  
      display.setCursor(3,3); //xpos,ypos
      display.println("eFish Farm Benin");
      display.println("by Bidossessi AGOSSOU");
       display.print("Temp:");
      display.print(temperature);
      display.print("C");
     
      display.print(", PH:");
      display.print(pH);
      display.print(", CO2:");
      display.print(CO2);
      display.print(", Turb:");  
      display.print(turbidity);
      display.print(", Electric:");
      display.print(EC);
      display.print(", tds:");
      display.print(TDS);
    
      display.display();
       */
        
 /*
  WiFiClient client = server.available();
  // wait for a client (web browser) to connect
  if (client)
  {
    Serial.println("\n[Client connected]");
    while (client.connected())
    {
      // read line by line what the client (web browser) is requesting
      if (client.available())
      {
        String line = client.readStringUntil('\r');
        Serial.print(line);
        // wait for end of client's request, that is marked with an empty line
        if (line.length() == 1 && line[0] == '\n')
        {
             String htmlPage =
            String("HTTP/1.1 200 OK\r\n") +
            "Content-Type: text/html\r\n" +
            "Connection: close\r\n" +  // the connection will be closed after completion of the response
            "Refresh: 5\r\n" +  // refresh the page automatically every 5 sec
            "\r\n" +
            "<!DOCTYPE HTML>" +
            "<html>" + output +
            "</html>" +
            "\r\n";
             client.println(htmlPage);
          break;
        }
      }
    }
   }
   */
       

       //send data to the Online Platforme at http:///
        sendDataToWebPage();
        
        //String fanStateStr,eggRotatorStateStr,windowsStateStr,humidifierStateStr,flameOdorStr;
        if(waterPumpState==1) waterPumpStateStr="water Pump ON";
        else waterPumpStateStr="water Pump OFF";
  
        if(oxygenPumpState==1) oxygenPumpStateStr="oxygen Pump ON";
        else oxygenPumpStateStr="oxygen Pump OFF";
  
         if(heatPumpState==1) heatPumpStateStr="heat Pump ON";
        else heatPumpStateStr="heat Pump OFF";
  
         if(colderPumpState==1) colderPumpStateStr="colder Pump ON";
        else colderPumpStateStr="colder Pump OFF";
  
         if(feederpumpState==1) feederpumpStateStr="colder Pump ON";
        else feederpumpStateStr="colder Pump OFF";
       //decision sripts end
    /*********************************/
    
        //send data to Thingspeak
         sendDataToThingSpeak();      
    
        //send data to MQTT host: test.mosquitto.org , topic: smartpoultry
        sendDataToMQTT();
  }
  
  
  delay(1);

}

//https://www.youtube.com/watch?v=6-RXqFS_UtU
//https://github.com/acrobotic/AI_Tips_ESP8266

//function to send data to thingspeak
void sendDataToThingSpeak(){
  
   if(client.connect(serverThinkspeak,80)) {
        String postStr = apiKey;
        
        postStr +="&field1=";
        postStr += String(temperature);
        
        postStr +="&field2=";
        postStr += String(O2);
        
        postStr +="&field3=";        
        postStr += String(NH3);
        
        postStr +="&field4=";
        postStr += String(TDS);
        
        postStr +="&field5=";
        postStr += String(EC);
        
        postStr +="&field6=";
        postStr += String(WaterLevel);
        
        postStr +="&field7=";
        postStr += String(CO2);
        
        postStr +="&field8=";
        postStr += String(turbidity);
        postStr += "\r\n\r\n"; 
          
        Serial.println("Sending data to Thingspeak");
        client.print("POST /update HTTP/1.1\n");
        client.print("Host: api.thingspeak.com\n");
        client.print("Connection: close\n");
        client.print("X-THINGSPEAKAPIKEY: "+apiKey+"\n");
        client.print("Content-Type: application/x-www-form-urlencoded\n");
        client.print("Content-Length: ");
        client.print(postStr.length());
        client.print("\n\n");
        client.print(postStr);
        Serial.println(postStr);
    }
    else{
      Serial.println("Fails to send data to Thingspeak");
    }
    //client.stop();
  }   



// function to send data to the online platform
//function to send data to online WebPage

void sendDataToWebPage(){   
      HTTPClient http;
    
    // Your Domain name with URL path or IP address with path
    http.begin(serverName);       
        // Specify content-type header
        http.addHeader("Content-Type", "application/x-www-form-urlencoded");
         while(!timeClient.update()) {
            timeClient.forceUpdate();
          }
            // The formattedDate comes with the following format:
          // 2018-05-28T16:00:13Z
          // We need to extract date and time
          formattedDate = timeClient.getFormattedDate();
          // Extract date
          int splitT = formattedDate.indexOf("T");
          dayStamp = formattedDate.substring(0, splitT);
        
          // Extract time
          timeStamp = formattedDate.substring(splitT+1, formattedDate.length()-1);
         
          dateTime=dayStamp+" "+timeStamp;
                                                                                                                      
          String httpRequestData = "api_key=" + apiKeyValue + "&UserID=" + UserID+"&SystemID=" + SystemID +"&location=" + location+"&dateTime=" + String(dateTime)+"&temperature=" + String(temperature)+"&O2=" +String(O2)+"&pH=" + String(pH)+"&turbidity=" +String(turbidity)+"&EC=" +String(EC)+"&CO2=" +String(CO2)+ "&WaterLevel=" + String(WaterLevel)+ "&NH3=" + String(NH3)+ "&TDS=" + String(TDS) + "&waterpump=" + String(waterPumpState) + "&oxygenpump=" + String(oxygenPumpState) + "&heatpump=" + String(heatPumpState) + "&colderpump=" + String(colderPumpState)+ "&feederpump=" + String(feederpumpState)+ " ";

           Serial.print("httpRequestData: ");
          Serial.println(httpRequestData);
          // Send HTTP POST request
          int httpResponseCode = http.POST(httpRequestData);
                    
          if (httpResponseCode>0) {
            Serial.print("HTTP Response code: ");
            Serial.println(httpResponseCode);
            Serial.print("Data sent to the database");
          }
          else {
            Serial.print("Error code: ");
            Serial.println(httpResponseCode);
            Serial.print("Data not sent to the database");
          }
          // Free resources
          http.end();
         
         
}

    
// function for remote control of pumps
 void remoteControle(){  
    const String url = "http://efishfarm.atwebpages.com/pumpWaterHeaterColder.txt";
    String payload ="";
    HTTPClient http1;  
    http1.begin(url);         
        // Specify content-type header
        http1.addHeader("Content-Type", "application/x-www-form-urlencoded");
        int httpCode1 = http1.GET();  //Make the request 
    if (httpCode1 > 0) { //Check for the returning code 
         payload = http1.getString();
        Serial.println(httpCode1);
        Serial.println(payload);      
        if(payload.endsWith("HEAT_EAU__ON")) heatPumpState=1;                             
        else if(payload.endsWith("HEAT_EAU__OFF")) heatPumpState=0;
        
        else if(payload.endsWith("EAU_ON")) heatPumpState=0;
        else if(payload.endsWith("EAU_OFF")) heatPumpState=0;
        
        else if(payload.endsWith("O2_ON")) oxygenPumpState=1;
        else if(payload.endsWith("O2_OFF")) oxygenPumpState=0;
        
        else if(payload.endsWith("COLD_EAU__ON")) colderPumpState=1;
        else if(payload.endsWith("COLD_EAU__OFF")) colderPumpState=0;

       else if(payload.endsWith("FEED_ON")) feederpumpState=1;
       else if(payload.endsWith("FEED_OFF")) feederpumpState=0;  
        
        else{  }  
       
    }
    else {
      Serial.println("Error on HTTP request");
    } 
    http1.end(); //Free the resources
 }


 
//function for MQTT
void sendDataToMQTT() {  
  WiFiClient MQTTclient;
  PubSubClient client(MQTTclient);
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");    
    client.setServer(mqtt_server, mqtt_port);   
    // Create a random client ID
    String clientId = "19870321";
    clientId += String(random(0xffff), HEX);
    // Attempt to connect
    if (client.connect(clientId.c_str(),MQTT_USER,MQTT_PASSWORD)) {
      Serial.println("\n Successfully connected to MQTT");
      //Once connected, publish an announcement...
       client.publish("eFishFarm", "Hello from Smart Fish Farming");
               
      String temperatureStr=String(temperature);
      String pHStr=String(pH);
      String O2Str=String(O2);
      String turbidityStr=String(turbidity);
      String NH3Str=String(NH3);
      String TDSStr=String(TDS);
      String ECStr=String(EC);
      String WaterLevelStr=String(WaterLevel);
      String CO2Str=String(CO2);
            
      char temperature_buff[20],pH_buff[20],O2_buff[20],turbidity_buff[20],NH3_buff[20],TDS_buff[20],EC_buff[20],WaterLevel_buff[20],CO2_buff[20],waterPumpState_buff[20],oxygenPumpState_buff[20],heatPumpState_buff[20],colderPumpState_buff[20],feederpumpState_buff[20];
      
       temperatureStr.toCharArray(temperature_buff, temperatureStr.length()+1);
       pHStr.toCharArray(pH_buff, pHStr.length()+1);
       O2Str.toCharArray(O2_buff, O2Str.length()+1);
       turbidityStr.toCharArray(turbidity_buff, turbidityStr.length()+1);
       NH3Str.toCharArray(NH3_buff, NH3Str.length()+1);
      TDSStr.toCharArray(TDS_buff, TDSStr.length()+1);
      ECStr.toCharArray(EC_buff, ECStr.length()+1);
       WaterLevelStr.toCharArray(WaterLevel_buff, WaterLevelStr.length()+1);
      CO2Str.toCharArray(CO2_buff,CO2Str.length()+1);
       waterPumpStateStr.toCharArray(WaterLevel_buff, waterPumpStateStr.length()+1);
       WaterLevelStr.toCharArray(waterPumpState_buff, WaterLevelStr.length()+1);
       oxygenPumpStateStr.toCharArray(oxygenPumpState_buff, oxygenPumpStateStr.length()+1);
       heatPumpStateStr.toCharArray(heatPumpState_buff, heatPumpStateStr.length()+1);
       colderPumpStateStr.toCharArray(colderPumpState_buff, colderPumpStateStr.length()+1);
      feederpumpStateStr.toCharArray(feederpumpState_buff, feederpumpStateStr.length()+1);       
      client.publish("eFishFarm", "Hello From Smart poultry");      
       client.publish("eFishFarm/temperature", temperature_buff);
       client.publish("eFishFarm/pH", pH_buff);
       client.publish("eFishFarm/O2", O2_buff);
       client.publish("eFishFarm/NH3", NH3_buff);
       client.publish("eFishFarm/TDS", TDS_buff);
       client.publish("eFishFarm/EC", EC_buff);
       client.publish("eFishFarm/WaterLevel", WaterLevel_buff);
       client.publish("eFishFarm/CO2", CO2_buff);
       client.publish("eFishFarm/waterPump", waterPumpState_buff);
       client.publish("eFishFarm/oxygenPump", oxygenPumpState_buff);
       client.publish("eFishFarm/heatPump", heatPumpState_buff);
       client.publish("eFishFarm/colderPump", colderPumpState_buff);
       client.publish("eFishFarm/feederPump",feederpumpState_buff);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      // Wait 5 seconds before retrying
      //delay(50000);
    }
  }
}
