import processing.serial.*;

Serial port;

void setup()
{
  size(300, 300);
  
  println("Bluetooth Car Control");
  println("wasd for movement, space to stop");
  
  port = new Serial(this, "/dev/cu.HC-06-DevB", 9600);
}

void draw()
{
  if (keyPressed)
  {
    println("Key pressed: " + keyCode);
  
    if (key == CODED)
    {
      if (keyCode == UP) port.write('w');
      else if (keyCode == DOWN) port.write('s');
      if (keyCode == LEFT) port.write('a');
      else if (keyCode == RIGHT) port.write('d');
    }
    if (key == ' ') port.write(' ');
  }
  
  else
  {
    port.write(0);
    println("CLEAR");
  }
}
