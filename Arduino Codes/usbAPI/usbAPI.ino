#include <SPI.h>
#include "USB.h"
#include "WiFi.h"
#include <esp_wifi.h>

#define V1 32
#define V2 33
#define V3 25

#define HSPI_MISO 19
#define HSPI_MOSI 13
#define HSPI_SCLK 18
#define HSPI_SS 5
#define BUFF_SIZE 10000



SPIClass * hspi = NULL;
unsigned long time_now;
bool adcEnable = false;
int long ReadData = 0;
int long MyData[2];
float adc_value;
int long CHANNEL=1;
int long TEMP=0;
int RFSW[8][3] = {{0, 0, 0}, 
                  {0, 0, 1}, 
                  {0, 1, 0}, 
                  {0, 1, 1}, 
                  {1, 0, 0}, 
                  {1, 0, 1}, 
                  {1, 1, 0}, 
                  {1, 1, 1}};

bool readBuffFlag=false;
// String readADCBuff="";
float readADCBuff[BUFF_SIZE];
unsigned int curBuffSize=0;
bool adcPlotter = false;


void RFswitch(int ch) {
  digitalWrite(V1, RFSW[ch][0]);
  digitalWrite(V2, RFSW[ch][1]);
  digitalWrite(V3, RFSW[ch][2]);
}

float GetAdcValue(){
  time_now = micros();
  hspi->beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));
  digitalWrite(HSPI_SS, LOW);
  while (micros() - time_now < round(1000000 * (1.0 / 1000))) {}
  ReadData = hspi->transfer(0);
  MyData[0] = ReadData;
  ReadData = hspi->transfer(0);
  MyData[1] = ReadData;
  digitalWrite(HSPI_SS, HIGH);
  hspi->endTransaction();
  adc_value = MyData[1] + (MyData[0] * 256);
  adc_value = adc_value * (2.5 / 65535.0) * 1000.0;
  return adc_value;
}

void setup() {
  // WiFi.mode(WIFI_MODE_APSTA);

  pinMode(V1, OUTPUT);
  pinMode(V2, OUTPUT);
  pinMode(V3, OUTPUT);

  digitalWrite(V1, LOW);
  digitalWrite(V2, LOW);
  digitalWrite(V3, LOW);

  hspi = new SPIClass(HSPI);
  hspi->begin(HSPI_SCLK, HSPI_MISO, HSPI_MOSI, HSPI_SS);
  pinMode(HSPI_SS, OUTPUT);
  digitalWrite(HSPI_SS, HIGH);

  RFswitch(CHANNEL);

  for (int i=0; i<BUFF_SIZE; ++i) {
    readADCBuff[i]=0;
  }
  
  Serial.begin(115200);
}

void loop() {
  if (readBuffFlag==true) {
    // readADCBuff.concat(String(GetAdcValue(), 2));
    // readADCBuff.concat(",");
    if (curBuffSize<BUFF_SIZE) {
      readADCBuff[curBuffSize]=GetAdcValue();
      curBuffSize+=1;
    }
  }

  if (Serial.available() > 0) {
    
    String rev = Serial.readString();
    String cmd = rev.substring(0, 3);
    cmd.toLowerCase();

    if (cmd == "ch_") {

      // TODO: check if readbuff flag is on, do no change phase.
      
      int ch = rev.substring(3, 4).toInt() - 1;
      RFswitch(ch);
      CHANNEL=ch;
      char printf[50];
      sprintf(printf,"ch: %d, ok", ch+1);
      Serial.println(printf);
    
    } 
    
    else if (cmd == "adc") {
      
      int sw = rev.substring(3, 4).toInt();
      
      if (sw == 0) {
        adcEnable = false;
      } 
      else if (sw == 1) {
        adcEnable = true;
      } 
      else if (sw == 2||sw == 3||sw == 4||sw == 5) {
        String str = "";
        for(int i = 1; i < sw*100; i++ ){
            str.concat(String(GetAdcValue(), 2));
            str.concat(",");
        }
        Serial.println(str);
      }
      else {
        adcEnable = false;
      }
    }

    else if (cmd=="rdb") {
      // ReadBegin: Start reading from ADC and storing into a buffer
      int ch = 1;
      RFswitch(ch);
      readBuffFlag=true;
      Serial.println("rdb");
    }

    else if (cmd=="rds"){
      // ReadStop: Stop reading and send the read buffer 
      readBuffFlag=false;
      
      // Old, when buf is string
      // int buff_len=readADCBuff.length();
      // Serial.println(buff_len);
      // Serial.println(readADCBuff);

      Serial.print(curBuffSize);
      Serial.println(",");
      for (int i=0; i<curBuffSize; ++i) {
        Serial.print(readADCBuff[i],2);
        Serial.println(",");
      }
      Serial.println("end");
      
      // resetting buffer
      // readADCBuff="";
      for (int i=0; i<BUFF_SIZE; ++i) {
        readADCBuff[i]=0;
      }
      curBuffSize=0;
  
    }

    else if (cmd == "spl") {
      // set phase 2
      RFswitch(1);
      // start adc plotter
      adcPlotter=true;
      
    }

    else if (cmd == "epl") {
      // end adc plotter
      adcPlotter=false;
    }

    else if (cmd=="mpp") {
      int channels[]= {1,3,4,6,7,8};
      for (int mpp_idx=0; mpp_idx<5; mpp_idx++){
        // Do one complete MPP
        for (int ch_idx=0; ch_idx<6; ch_idx++){
          int ch = channels[ch_idx] - 1;
          RFswitch(ch);
          CHANNEL=ch;
          delay(10);
        }
      }
      
      Serial.println("mpp");
    
    }
    
    else if (cmd == "mac"){
      Serial.println(WiFi.macAddress());
      // Serial.println(TEMP);
      // TEMP+=1;
    }
    else {
      Serial.println("cmd: not found");
    }
  }

  if (adcPlotter==true){
    Serial.println(GetAdcValue()); 
    delay(50);
  }
}