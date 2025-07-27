// index.js
const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const { spawn } = require('child_process');

const app = express();
app.use(cors());
app.use(bodyParser.json());

app.post('/chat', async (req, res) => {
  const { message } = req.body;

  const ollama = spawn('ollama', ['run', 'mistral']);

  let response = '';
  ollama.stdout.on('data', (data) => {
    response += data.toString();
  });

  ollama.stdin.write(`${message}\n`);
  ollama.stdin.end();

  ollama.on('close', () => {
    res.json({ reply: response.trim() });
  });
});

app.listen(3001, () => {
  console.log('Server listening on http://localhost:3001');
});

