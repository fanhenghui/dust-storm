var app = require('express')();
var path = require('path');
var http = require('http').Server(app);
var io = require('socket.io')(http);
var util = require('util');
var mongoose = require('mongoose');
var session = require('express-session');
var logger = require('morgan');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');

var routes = require('./public/routes/index');
global.userList = require('./public/routes/logged-user-list');
//var log4js = require('./public/log/log').log4js();

// log4js.configure({
//   appenders: [
//     {type: 'concole'},
//     {
//       type: 'file',
//       filename: 'log/logs//server.log',
//       maxLogSize: 1024,
//       backup:3,
//       category: 'normal'
//     }
//   ],
//   replaceConsole: true
// });

// log4js.configure({
//   appenders: {
//     out: { type: 'stdout' },
//     app: { type: 'file', filename: 'application.log' }
//   },
//   categories: {
//     default: { appenders: [ 'out', 'app' ], level: 'debug' }
//   }
// });
global.appPort = 8000;
if (process.argv.length == 3)  {
    global.appPort = process.argv[2];
}
console.log('webviewer port : ' + appPort);

global.dbHandel = require('./public/database/dbhandel');
global.db = mongoose.connect('mongodb://localhost:27017/nodedb');
global.db.connection.on('error', function(error) {
    console.log('connect user mongon DB failed：' + error);
});
global.db.connection.on('open', function() {
    console.log('connect user mongon DB success.')
});

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
app.set('views', path.join(__dirname, '/public/views'));
app.engine('html', require('ejs').renderFile);
app.set('view engine', 'html');
app.use('/', routes);

//app.use(log4js.connectLogger(this.logger('normal') , {level:'auto', format:':method :url '}));
//connect to web socket
io.set('log level',0);//好像没有设置成功
io.on('connection', require('./public/be/be_proxy').onIOSocketConnect);

//process quit callback
process.on('exit', function(err) {
    console.log('process exit.');
    require('./public/be/be_proxy').cleanIOSocketConnect();
});
process.on('uncaughtException', function(err) {
    console.error('An uncaught error occurred!');
    process.exit(99);
});
process.on('SIGINT', function(err) {
    console.log('catches ctrl+c event.');
    process.exit(2);
});

var server = http.listen(global.appPort, function() {
    var address = server.address();
    console.log('address is ', util.inspect(address));
});