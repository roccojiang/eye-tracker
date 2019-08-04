// Motor A (drive)

int enA = 9;
int in1 = 8;
int in2 = 7;

// Motor B (steering)

int enB = 3;
int in3 = 5;
int in4 = 4;

void setup()

{

  // Set all the motor control pins to outputs

  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);

  // Initialise serial
  Serial.begin(9600);

}

void drive(int dir, int spd)
{
  // Direction: 1 (forward), 0 (backward)
  // Speed: ranges from 0~255
  
  // Turn on motor A (drive)
  digitalWrite(in1, dir == 1 ? LOW : HIGH);
  digitalWrite(in2, dir == 1 ? HIGH : LOW);

  // Set speed
  analogWrite(enA, spd);
}

void turn(int dir, int spd)
{
  // Direction: 1 (right), 0 (left)
  // Speed: ranges from 0~255 (recommended max 255)

  // Turn on motor B (steering)
  digitalWrite(in3, dir == 1 ? LOW : HIGH);
  digitalWrite(in4, dir == 1 ? HIGH : LOW);

  // Set speed
  analogWrite(enB, spd);
}

void stop()
{
  // Turn off all motors
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
}

void demoOne()

{

  // This function will run the motors in both directions at a fixed speed

  // Turn on motor A (drive)

  digitalWrite(in1, HIGH);
  digitalWrite(in2, LOW);

  // Set speed to 200 out of possible range 0~255

  analogWrite(enA, 200);

  // Turn on motor B (steering)

  digitalWrite(in3, HIGH);
  digitalWrite(in4, LOW);

  // Set speed to 255 out of possible range 0~255

  analogWrite(enB, 255);

  delay(2000);

  // Now change motor directions

  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);  
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH); 

  delay(2000);

  // Now turn off motors

  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);  
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);

}

void demoTwo()

{

  // This function will run the motors across the range of possible speeds
  // Note that maximum speed is determined by the motor itself and the operating voltage

  // Turn on motors

  digitalWrite(in1, LOW);
  digitalWrite(in2, HIGH);  
  digitalWrite(in3, LOW);
  digitalWrite(in4, HIGH); 

  // Accelerate from zero to maximum speed

  for (int i = 0; i < 256; i++)

  {

    analogWrite(enA, i);
    analogWrite(enB, i);

    delay(20);

  } 

  // Decelerate from maximum speed to zero

  for (int i = 255; i >= 0; --i)

  {

    analogWrite(enA, i);
    analogWrite(enB, i);

    delay(20);

  } 

  // Now turn off motors

  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);  
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);  

}

void demoBluetooth()
{
  // Quick, dirty code to test out bluetooth communication
  // Can only be using one motor at a time
  
  while (Serial.available())
  {
    char input = (char) Serial.read();
    
    switch(input)
    {
      case 'f':
        Serial.println("Driving forward");
        drive(1, 150);
        break;

      case 'b':
        Serial.println("Driving backward");
        drive(0, 150);
        break;

      case 'r':
        Serial.println("Turning right");
        turn(1, 255);
        break;

      case 'l':
        Serial.println("Turning left");
        turn(0, 255);
        break;
      
      case 's':
        Serial.println("Stopping");
        stop();
        break;
    }
  }
}

void loop()

{

  demoBluetooth();

}
