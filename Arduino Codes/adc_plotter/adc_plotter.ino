#include <SPI.h>
#include "freertos/queue.h"
#include "USB.h"
#include "WiFi.h"
#define V1 32
#define V2 33
#define V3 25

#define HSPI_MISO 19
#define HSPI_MOSI 13
#define HSPI_SCLK 18
#define HSPI_SS 5

SPIClass * hspi = NULL;
unsigned long time_now;
bool adcEnable = false;
int long ReadData = 0;
int long MyData[2];
float adc_value;


int RFSW[8][3] = {{0, 0, 0}, 
                  {0, 0, 1}, 
                  {0, 1, 0}, 
                  {0, 1, 1}, 
                  {1, 0, 0}, 
                  {1, 0, 1}, 
                  {1, 1, 0}, 
                  {1, 1, 1}};

void RFswitch(int ch) {
  digitalWrite(V1, RFSW[ch][0]);
  digitalWrite(V2, RFSW[ch][1]);
  digitalWrite(V3, RFSW[ch][2]);
}

float GetAdcValue(){
  time_now = micros();
  hspi->beginTransaction(SPISettings(10000000, MSBFIRST, SPI_MODE1));
  digitalWrite(HSPI_SS, LOW);
  //while (micros() - time_now < round(1000000 * (1.0 / 1000))) {}
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
  pinMode(V1, OUTPUT);
  pinMode(V2, OUTPUT);
  pinMode(V3, OUTPUT);

  digitalWrite(V1, LOW);
  digitalWrite(V2, LOW);
  digitalWrite(V3, LOW);
  Serial.println("hola, task!\n");
  hspi = new SPIClass(HSPI);
  hspi->begin(HSPI_SCLK, HSPI_MISO, HSPI_MOSI, HSPI_SS);
  pinMode(HSPI_SS, OUTPUT);
  digitalWrite(HSPI_SS, HIGH);
  RFswitch(1);
  Serial.begin(9600);
}

void loop() {
  Serial.println(GetAdcValue()); 
  delay(50);
}