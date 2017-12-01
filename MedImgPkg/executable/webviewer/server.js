const app = require('express')();
const path = require('path');
const httpServer = require('http').Server(app);
const wsServer = require('socket.io')(httpServer);
const util = require('util');
const session = require('express-session');
const logger = require('morgan');
const cookieParser = require('cookie-parser');
const bodyParser = require('body-parser');
const fs = require('fs');

global.appPort = 8000;
if (process.argv.length == 3)  {
    global.appPort = process.argv[2];
}

global.subPage = '';
if (process.argv.length == 4) {
    global.appPort = process.argv[2];
    global.subPage = process.argv[3];
}
console.log('webviewer port : ' + appPort);
console.log('webviewer subpage : ' + subPage);

//get config path
let confitPathFile = path.join(__dirname,'public','config','config_path');
let configPathData = fs.readFileSync(confitPathFile);
let linesConfigPath = configPathData.toString().split('\n');
if (linesConfigPath.length == 0) {
    console.log('get config path failed!');
    throw('get config path failed!');
}
global.configPath = linesConfigPath[0];
global.dbHandel = require('./public/database/db-handel');

let routes = require('./public/routes/index');
// use session for login
app.use(session({
    secret: 'secret',
    cookie: {
        maxAge: 1000 * 60 * 30
    }
})); // 30 min timeout
app.use(logger('dev'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({
    extended: true
}));
app.use(cookieParser());
app.use(require('express').static(__dirname + '/public'));

// how to render template files
app.set('views', path.join(__dirname, '/public' + global.subPage + '/views'));
app.engine('html', require('ejs').renderFile);
app.set('view engine', 'html');
app.use('/', routes);

//app.use(log4js.connectLogger(this.logger('normal') , {level:'auto', format:':method :url '}));
//connect to web socket
wsServer.set('log level',0);//好像没有设置成功
wsServer.on('connection', require('./public/be/be-proxy').onWebSocketConnect);

//process quit callback
process.on('exit', (err)=> {
    global.dbHandel.closeDB();
    console.log('process exit.');
    require('./public/be/be-proxy').cleanWebSocketConnection();
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

httpServer.listen(global.appPort);