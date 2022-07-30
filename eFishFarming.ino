
/******** end FOR LCD Display*********/
/*
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
*/
/******** end FOR LCD Display*********/


#include <OneWire.h>  // for DS18b20 temperature sensor
#include <DallasTemperature.h>  // for DS18b20 temperature sensor

int  waterPumpState=0,oxygenPumpState=0,heatPumpState=0,colderPumpState=0,feederPumpState=0;
 /** Ammonia NH3 sensor MQ 137 */
  #define RL 47  //The value of resistor RL is 47K by default
  #define m -0.263 //Enter calculated Slope 
  #define b 0.42 //Enter calculated intercept
  #define Ro 20 //Enter found Ro value
  #define MQ_sensor A0 //Sensor is connected to A0
 /*********************************/

  /** SEN0189 turbidity sensor*/
  #define Turbidity_SENSOR A1 //Sensor is connected to A1
  /*********************************/

 /** DS 18b20 temperature sensor*/
    // Data wire is plugged into pin 2 on the Arduino
    #define ONE_WIRE_BUS 2
     
    // Setup a oneWire instance to communicate with any OneWire devices 
    // (not just Maxim/Dallas temperature ICs)
    OneWire oneWire(ONE_WIRE_BUS);
     
    // Pass our oneWire reference to Dallas Temperature.
    DallasTemperature sensors(&oneWire);
 /*********************************/

 /** IR Distance Sensor with Cables - Sharp Infrared Sensor Module GP2Y0A02YK0F.  measures the distance between 4cm to 180cm*/
    #include <SharpIR.h>
    
    #define IR A2 // define signal pin
    #define model 1080 // used 1080 because model GP2Y0A21YK0F is used
    SharpIR SharpIR(IR, model);

 /*********************************/

  /** Electrical conductivity sensor*/
    #include "DFRobot_EC.h"
    #include <EEPROM.h>     
    #define EC_PIN A3
    float thethevoltage,ecValue;
    DFRobot_EC ec;
  /*********************************/

/** TDS sensor*/
     #define TdsSensorPin A4
    #define VREF 5.0      // analog reference voltage(Volt) of the ADC
    #define SCOUNT  30           // sum of sample point
    int analogBuffer[SCOUNT];    // store the analog value in the array, read from ADC
    int analogBufferTemp[SCOUNT];
    int analogBufferIndex = 0,copyIndex = 0;
    float averageVoltage = 0,tdsValue = 0;

/*********************************/

/** CO2 sensor*/
  #include "CO2Sensor.h"
  CO2Sensor co2Sensor(3, 0.99, 100);
/*********************************/

/*********************************/

/** Dissolved Oxygen sensor*/
    #include <SoftwareSerial.h>                           //we have to include the SoftwareSerial library, or else we can't use it
    #define rx 4                                          //define what pin rx is going to be
    #define tx 5                                           //define what pin tx is going to be    
    SoftwareSerial myserial(rx, tx);                      //define how the soft serial port is going to work   
    
    String inputstring = "";                              //a string to hold incoming data from the PC
    String sensorstring = "";                             //a string to hold the data from the Atlas Scientific product
    boolean input_string_complete = false;                //have we received all the data from the PC
    boolean sensor_string_complete = false;               //have we received all the data from the Atlas Scientific product
    float DO;                                             //used to hold a floating point number that is the DO

/*********************************/
/******** FOR THE Ph*********/
#define SensorPin A5            //pH meter Analog output to Arduino Analog Input 0
#define Offset 0.00            //deviation compensate
#define LED 13
#define samplingInterval 20
#define printInterval 800
#define ArrayLenth  40    //times of collection
int pHArray[ArrayLenth];   //Store the average value of the sensor feedback
int pHArrayIndex=0;
float pH;
/******** end FOR Ph*********/

float temperature1=13.00;


void setup(void)
{
 
  Serial.begin(9600); // start serial port
/*
  // initialize with the I2C addr 0x3C
        if(!display.begin(SSD1306_SWITCHCAPVCC)){
            Serial.println("SSD1306 allocation failed");
            for(;;);//Don't proceed, loop forever
        }     
*/
      
  //Serial.println("Dallas Temperature IC Control Library Demo"); // DS 18b20 temperature sensor
  // Start up the library
  sensors.begin();// DS 18b20 temperature sensor
   ec.begin();// Electrical conductivity sensor begins 
    pinMode(TdsSensorPin,INPUT);// for TDS sensor

     myserial.begin(9600);  //For the DO, set baud rate for the software serial port to 9600
      inputstring.reserve(10);//For the DO, set aside some bytes for receiving data from the PC
     sensorstring.reserve(30);  //For the DO   
  
}

 
 void serialEvent() {                                  //if the hardware serial port_0 receives a char
  inputstring = Serial.readStringUntil(13);           //read the string until we see a <CR>
  input_string_complete = true;                       //set the flag used to tell if we have received a completed string from the PC
}


void loop(void)
{  
     
     /*
     // display on LCD     
      display.setTextSize(1); // Normal 1:1 pixel
      display.setTextColor(WHITE); // Draw white text
       display.clearDisplay(); // clear display  
      display.setCursor(3,3); //xpos,ypos
      display.println("eFish Farm Benin");
      display.println("");
      temperature1=temperature1+1;
      display.println("Emmanuel AGOSSOU");      
      display.println("");
      display.print("Temp:");
      display.print(temperature1);
      display.print("C");
    
      display.display();
  */
     
      /*************for pH ********************/
        static unsigned long samplingTime = millis();
      static unsigned long printTime = millis();
      static float pHValue,voltage;
      if(millis()-samplingTime > samplingInterval)
      {
          pHArray[pHArrayIndex++]=analogRead(SensorPin);
          if(pHArrayIndex==ArrayLenth)pHArrayIndex=0;
          voltage = avergearray(pHArray, ArrayLenth)*5.0/1024;
          pHValue = 3.5*voltage+Offset;
          samplingTime=millis();
      }
      
   /*********************************/ 
      
      
      sensors.requestTemperatures(); // Send the command to get temperatures
       //Serial.print("Temperature is: ");
      float temperature = sensors.getTempCByIndex(0);        
      //Serial.print(sensors.getTempCByIndex(0)); // Why "byIndex"? 
        // You can have more than one IC on the same bus. 
        // 0 refers to the first IC on the wire
     //Serial.println(" °C");
          /*********************************/    

      
      
   /** Dissolved Oxygen sensor*/
         if (input_string_complete == true) {                //if a string from the PC has been received in its entirety
            myserial.print(inputstring);                      //send that string to the Atlas Scientific product
            myserial.print('\r');                             //add a <CR> to the end of the string
            inputstring = "";                                 //clear the string
            input_string_complete = false;                    //reset the flag used to tell if we have received a completed string from the PC
        }
    
        if (myserial.available() > 0) {                     //if we see that the Atlas Scientific product has sent a character
          char inchar = (char)myserial.read();              //get the char we just received
          sensorstring += inchar;                           //add the char to the var called sensorstring
          if (inchar == '\r') {                             //if the incoming character is a <CR>
            sensor_string_complete = true;                  //set the flag
          }
        }
      if (sensor_string_complete == true) {               //if a string from the Atlas Scientific product has been received in its entirety
         //Serial.print("DO: "); 
        //Serial.println(sensorstring);                     //send that string to the PC's serial monitor
                                                     //uncomment this section to see how to convert the DO reading from a string to a float 
        if (isdigit(sensorstring[0])) {                   //if the first character in the string is a digit
          DO = sensorstring.toFloat();
          //Serial.println(DO);    //convert the string to a floating point number so it can be evaluated by the Arduino
          /*
          if (DO >= 6.0) {                                //if the DO is greater than or equal to 6.0
            Serial.println("high");                       //print "high" this is demonstrating that the Arduino is evaluating the DO as a number and not as a string
          }
          if (DO <= 5.99) {                               //if the DO is less than or equal to 5.99
            Serial.println("low");                        //print "low" this is demonstrating that the Arduino is evaluating the DO as a number and not as a string
          }*/
          
        }
        sensorstring = "";                                //clear the string
        sensor_string_complete = false;                   //reset the flag used to tell if we have received a completed string from the Atlas Scientific product
      }
  /*********************************/

 
     
     /** Turbidity sensor SEN0189 */
      float thevoltage=0.004888*analogRead(Turbidity_SENSOR);  //in V
      float turbidity=-1120.4*thevoltage*thevoltage+5742.3*thevoltage-4352.9;  //in NTU
     
    /*********************************/
    
   /** Electrical conductivity sensor*/
      static unsigned long timepoint = millis();
    if(millis()-timepoint>1000U)  //time interval: 1s
    {
      timepoint = millis();
      thethevoltage = analogRead(EC_PIN)/1024.0*5000;   // read the voltage
      //temperature = readTemperature();          // read your temperature sensor to execute temperature compensation
      ecValue =  ec.readEC(thethevoltage,temperature);  // convert voltage to EC with temperature compensation
      // Serial.print("^C  EC:");
      //Serial.print(ecValue,2);
     // Serial.println("ms/cm");
    }
    ec.calibration(thethevoltage,temperature);          // calibration process by Serail CMD
   /*********************************/

/** CO2 sensor*/
  co2Sensor.calibrate();
  int CO2val = co2Sensor.read();
  //Serial.print("CO2 value: ");
  //Serial.print(CO2val);
  //Serial.println("ppm");  //ppm or mg/L
  //if(CO2val> 1000)  Serial.println("CO2 exceeds safe concentration of 1000ppm, please ventilate");
   /*********************************/
   
    /** IR Distance Sensor with Cables - Sharp Infrared Sensor Module GP2Y0A02YK0F.  measures the distance between 4cm to 180cm*/
      int dis=SharpIR.distance();  // this returns the distance to the object you're measuring    
      //Serial.print("Water level(cm): ");  // returns it to the serial monitor
      //Serial.println(dis);
   /*********************************/

 /** Ammonia NH3 sensor MQ 137 */
      float VRL; //Voltage drop across the MQ sensor
      float Rs; //Sensor resistance at gas concentration 
      float ratio; //Define variable for ratio
       
      VRL = analogRead(MQ_sensor)*(5.0/1023.0); //Measure the voltage drop and convert to 0-5V
      Rs = ((5.0*RL)/VRL)-RL; //Use formula to get Rs value
      ratio = Rs/Ro;  // find ratio Rs/Ro
      float NH3ppm = pow(10, ((log10(ratio)-b)/m)); //use formula to calculate ppm for Ammonia NH3
      //Serial.print("Ammonia NH3 (ppm): ");
      //Serial.println(NH3ppm);
     /*********************************/
    
   /** TDS sensor*/

          static unsigned long analogSampleTimepoint = millis();
         if(millis()-analogSampleTimepoint > 40U)     //every 40 milliseconds,read the analog value from the ADC
         {
           analogSampleTimepoint = millis();
           analogBuffer[analogBufferIndex] = analogRead(TdsSensorPin);    //read the analog value and store into the buffer
           analogBufferIndex++;
           if(analogBufferIndex == SCOUNT) 
               analogBufferIndex = 0;
         }   
         static unsigned long printTimepoint = millis();
         if(millis()-printTimepoint > 800U)
         {
            printTimepoint = millis();
            for(copyIndex=0;copyIndex<SCOUNT;copyIndex++)
              analogBufferTemp[copyIndex]= analogBuffer[copyIndex];
            averageVoltage = getMedianNum(analogBufferTemp,SCOUNT) * (float)VREF / 1024.0; // read the analog value more stable by the median filtering algorithm, and convert to voltage value
            float compensationCoefficient=1.0+0.02*(temperature-25.0);    //temperature compensation formula: fFinalResult(25^C) = fFinalResult(current)/(1.0+0.02*(fTP-25.0));
            float compensationVolatge=averageVoltage/compensationCoefficient;  //temperature compensation
            tdsValue=(133.42*compensationVolatge*compensationVolatge*compensationVolatge - 255.86*compensationVolatge*compensationVolatge + 857.39*compensationVolatge)*0.5; //convert voltage value to tds value
            //Serial.print("voltage:");
            //Serial.print(averageVoltage,2);
            //Serial.print("V   ");
            //Serial.print("TDS Value:");
            //Serial.print(tdsValue,0);
            //Serial.println("ppm");
         }
      
 
      // preparing data to send to serial monitor
      ecValue=tdsValue*2;
      String msg=String(temperature);
      msg+=" "+String(DO);
      msg+=" "+String(turbidity);          
      msg+=" "+String(ecValue);
      msg+=" "+String(CO2val);
      msg+=" "+String(dis);
      msg+=" "+String(NH3ppm);
      msg+=" "+String(tdsValue); 
      msg+=" "+String(pHValue); 

      msg+=" "+String(waterPumpState); 
      msg+=" "+String(oxygenPumpState); 
      msg+=" "+String(heatPumpState); 
      msg+=" "+String(colderPumpState); 
      msg+=" "+String(feederPumpState); 
      
      
     
  //Serial.println("**************************************************************************");
   Serial.println(msg);// send data to serial monitor for esp

  
      delay(3000);
}


// funtion utile pour le calcul du TDS
int getMedianNum(int bArray[], int iFilterLen) 
{
      int bTab[iFilterLen];
      for (byte i = 0; i<iFilterLen; i++)
      bTab[i] = bArray[i];
      int i, j, bTemp;
      for (j = 0; j < iFilterLen - 1; j++) 
      {
      for (i = 0; i < iFilterLen - j - 1; i++) 
          {
        if (bTab[i] > bTab[i + 1]) 
            {
        bTemp = bTab[i];
            bTab[i] = bTab[i + 1];
        bTab[i + 1] = bTemp;
         }
      }
      }
      if ((iFilterLen & 1) > 0)
    bTemp = bTab[(iFilterLen - 1) / 2];
      else
    bTemp = (bTab[iFilterLen / 2] + bTab[iFilterLen / 2 - 1]) / 2;
      return bTemp;
}

// function utile pour le calcul du pH
double avergearray(int* arr, int number){
  int i;
  int max,min;
  double avg;
  long amount=0;
  if(number<=0){
    //Serial.println("Error number for the array to avraging!/n");
    return 0;
  }
  if(number<5){   //less than 5, calculated directly statistics
    for(i=0;i<number;i++){
      amount+=arr[i];
    }
    avg = amount/number;
    return avg;
  }else{
    if(arr[0]<arr[1]){
      min = arr[0];max=arr[1];
    }
    else{
      min=arr[1];max=arr[0];
    }
    for(i=2;i<number;i++){
      if(arr[i]<min){
        amount+=min;        //arr<min
        min=arr[i];
      }else {
        if(arr[i]>max){
          amount+=max;    //arr>max
          max=arr[i];
        }else{
          amount+=arr[i]; //min<=arr<=max
        }
      }//if
    }//for
    avg = (double)amount/(number-2);
  }//if
  return avg;
}
