const express = require('express');
const app = express();
const path = require('path');
const httpServer = require('http').Server(app);
const wsServer = require('socket.io')(httpServer);
const util = require('util');
const ejs = require('ejs');
const session = require('express-session');
const logger = require('morgan');
const cookieParser = require('cookie-parser');
const bodyParser = require('body-parser');
const fs = require('fs');
const be = require('./libs/be-proxy');
const config = require('./config/config');
const dbHandle = require('./libs/db-handle');

let appPort = 8000;
if (process.argv.length == 3)  {
    appPort = process.argv[2];
}
console.log('webviewer port : ' + appPort);

global.dbHandel = dbHandle;

app.use(logger('dev'));

//cookie & session
app.use(session({
    secret: 'secret',
    cookie: {maxAge: 1000 * 60 * 30}
})); // 30 min timeout

app.use(cookieParser());
app.use(express.static(__dirname + '/www'));

//post(URL-encoded)
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));

//post file 
//TODO 

//render template
app.set('views', path.join(__dirname, '/views'));
app.engine('html', ejs.renderFile);
app.set('view engine', 'html');

//router
app.use('/', require('./routes/index'));

//WebSocket Server
wsServer.set('log level',0);//好像没有设置成功
wsServer.on('connection', be.onWebSocketConnect);

//HTTP Server
httpServer.listen(appPort);

//process quit callback
process.on('exit', (err)=> {
    global.dbHandel.closeDB();
    console.log('process exit.');
    be.cleanWebSocketConnection();
});
process.on('uncaughtException', (err)=> {
    console.error('An uncaught error occurred!');
    console.log(err);
    process.exit(99);
});
process.on('SIGINT', (err)=> {
    console.log('catches ctrl+c event.');
    process.exit(2);
});