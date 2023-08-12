const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const operations = ['+', '-', '*', '/'];

function performOperation(operator, num1, num2) {
  switch (operator) {
    case '+':
      return num1 + num2;
    case '-':
      return num1 - num2;
    case '*':
      return num1 * num2;
    case '/':
      return num1 / num2;
    default:
      return 'Invalid operator';
  }
}

function startCalculator() {
  rl.question('Enter first number: ', num1 => {
    rl.question('Enter second number: ', num2 => {
      rl.question(`Enter operation (${operations.join(', ')}): `, operator => {
        if (operations.includes(operator)) {
          const result = performOperation(operator, parseFloat(num1), parseFloat(num2));
          console.log(`Result: ${result}`);
        } else {
          console.log('Invalid operator');
        }
        rl.close();
      });
    });
  });
}

startCalculator();
/**
 

const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const operations = ['add', 'subtract', 'multiply', 'divide'];

function performOperation(operator, num1, num2) {
  switch (operator) {
    case 'add':
      return num1 + num2;
    case 'subtract':
      return num1 - num2;
    case 'multiply':
      return num1 * num2;
    case 'divide':
      return num1 / num2;
    default:
      return 'Invalid operator';
  }
}

function startCalculator() {
  rl.question('Enter first number: ', num1 => {
    rl.question('Enter second number: ', num2 => {
      rl.question(`Enter operation (${operations.join(', ')}): `, operator => {
        if (operations.includes(operator)) {
          const result = performOperation(operator, parseFloat(num1), parseFloat(num2));
          console.log(`Result: ${result}`);
        } else {
          console.log('Invalid operator');
        }
        rl.close();
      });
    });
  });
}

startCalculator();




// Importing events
const EventEmitter = require('events');

// Initializing event emitter instances
var eventEmitter = new EventEmitter();

// Registering to myEvent
eventEmitter.on('myEvent', (msg) => {
console.log(msg);
});

// Triggering myEvent
eventEmitter.emit('myEvent', "First event");

 */