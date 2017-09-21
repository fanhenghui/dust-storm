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

global.dbHandel = require('./public/database/dbhandel');
global.db = mongoose.connect('mongodb://localhost:27017/nodedb');
global.db.connection.on("error", function (error) {
  console.log("connect user mongon DB failed：" + error);
});
global.db.connection.on("open", function () {
  console.log("connect user mongon DB success.")
});

// use session for login
app.use(session( {secret: 'secret', cookie: {maxAge: 1000 * 60 * 30}}));  // 30 min timeout
app.use(logger('dev'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));
app.use(cookieParser());
app.use(require('express').static(__dirname + '/public'));

// how to render template files
app.set('views', path.join(__dirname, '/public/views'));
app.engine('html', require('ejs').renderFile);
app.set('view engine', 'html');
app.use('/', routes);

//connect to web socket
io.on('connection', require('./public/be/be_proxy').onIOSocketConnect);

var server = http.listen(8000, function() {
  var address = server.address();
  console.log('address is ', util.inspect(address));
});