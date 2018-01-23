const express = require('express');
const app = express();
const httpServer = require('http').Server(app);
const wsServer = require('socket.io')(httpServer);
const ejs = require('ejs');
const sessionParser = require('cookie-session');
const cookieParser = require('cookie-parser');
const logger = require('morgan');
const bodyParser = require('body-parser');
const be = require('./libs/be-proxy');
const config = require('./config/config');
const dbHandle = require('./libs/db-handle');

let appPort = 8000;
if (process.argv.length == 3)  {
    appPort = process.argv[2];
}
console.log('server port : ' + appPort);

//create DB pool
dbHandle.createDB(global);

app.use((req,res,next)=>{
    req.db = global.db;
    next();
});

app.use(logger('dev'));

//cookie & session
app.use(cookieParser(require('./config/cookie-keys')));
app.use(sessionParser({
    keys:require('./config/session-keys'),
    maxAge: 30*60*1000//30min    
}));

//post(URL-encoded)
app.use(bodyParser.urlencoded({extended: true}));

//post file 
//TODO 

//render template
app.engine('html', ejs.renderFile);
app.set('view engine', 'ejs');
app.set('views', './views');

//router
app.use('/', require('./routers/index'));
app.use('/user', require('./routers/user'));
app.use('/app', require('./routers/app'));

//static file 
app.use(express.static('./www'));

//WebSocket Server
wsServer.on('connection', be.onWebSocketConnect);

//HTTP Server
httpServer.listen(appPort);

//process quit callback
process.on('exit', (err)=> {
    console.log('process exit.');
    be.cleanWebSocketConnection();

    //release DB
    dbHandle.releaseDB(global.db);
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