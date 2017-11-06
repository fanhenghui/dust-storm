var app = require('express')();
var path = require('path');
var http = require('http').Server(app);
var io = require('socket.io')(http);
var util = require('util');
var session = require('express-session');
var logger = require('morgan');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
var fs = require('fs');

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
var confitPathFile = path.join(__dirname,'public','config','config_path');
var configPathData = fs.readFileSync(confitPathFile);
var linesConfigPath = configPathData.toString().split('\n');
if (linesConfigPath.length == 0) {
    console.log('get config path failed!');
    throw('get config path failed!');
}
global.configPath = linesConfigPath[0];
global.dbHandel = require('./public/database/db-handel');

var routes = require('./public/routes/index');
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
io.set('log level',0);//好像没有设置成功
io.on('connection', require('./public/be/be-proxy').onIOSocketConnect);

//process quit callback
process.on('exit', function(err) {
    console.log('process exit.');
    require('./public/be/be-proxy').cleanIOSocketConnect();
});
process.on('uncaughtException', function(err) {
    console.error('An uncaught error occurred!');
    console.log(err);
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