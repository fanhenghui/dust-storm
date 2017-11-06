var mySQL = require('mysql');
var dbAdd = require("./db-address");
var fs = require('fs');

function getRemoteDBAddr() {
	let addr = {};
	let data = fs.readFileSync(global.configPath);
	let configStr = data.toString();
	let configLines = configStr.split('\n');
	if (configLines.length < 5) {
		console.log('config file damadged.')
		return {};
	}
	for (let i = 0; i< configLines.length; ++i) {
		let items = configLines[i].split(' ');
		if (items.length == 3 && items[0] == 'RemoteDBIP')	 {
			addr['ip'] = items[2];
		} else if (items.length == 3 && items[0] == 'RemoteDBPort')	 {
			addr['port'] = items[2];
		} else if (items.length == 3 && items[0] == 'RemoteDBUser')	 {
			addr['user'] = items[2];
		} else if (items.length == 3 && items[0] == 'RemoteDBPWD')	 {
			addr['password'] = items[2];
		} else if (items.length == 3 && items[0] == 'RemoteDBName')	 {
			addr['database'] = items[2];
		}

		if (i > 6 && addr['ip'] != undefined &&
			addr['port'] != undefined &&
			addr['user'] != undefined &&
			addr['password'] != undefined &&
			addr['database'] != undefined) {
			break;
		}
	}
	return addr;
}

module.exports = { 
	signIn: function(name, pwd, callback){
		let addr = getRemoteDBAddr();
		if ( !(addr['ip'] != undefined &&
		        addr['port'] != undefined &&
		        addr['user'] != undefined &&
		        addr['password'] != undefined &&
		        addr['database'] != undefined) ) {
		    callback(-3, 'database login message incompleted.');
		}
		var connection = mySQL.createConnection( {
			host:addr.ip,
			port:addr.port,
			user:addr.user,
			password:addr.password,
			database:addr.database,
		});

		connection.query('select * from usr where name="' + name +'"', function(err, res, fields) {
			if(err) {
				//数据库错误
				callback(-3,err);
				connection.end();
			} else if(res.length == 0) {
				//用户名不存在
				callback(-1);
				connection.end();
			} else {
				if (res[0].password == pwd ) {
					//登录成功
					callback(0);
					connection.end();
				} else {
					//密码不正确
					callback(-2);
					connection.end();
				} 
			}
		});
	},
	
	register: function(name, pwd, callback) {
		let addr = getRemoteDBAddr();
		if ( !(addr['ip'] != undefined &&
		        addr['port'] != undefined &&
		        addr['user'] != undefined &&
		        addr['password'] != undefined &&
		        addr['database'] != undefined) ) {
		    callback(-3, 'database login message incompleted.');
		}
		var connection = mySQL.createConnection( {
			host:addr.ip,
			port:addr.port,
			user:addr.user,
			password:addr.password,
			database:addr.database,
		});

		connection.query('select * from usr where name="' + name +'"', function(err, res, fields) {
			if(err) {
				//数据库错误
				callback(-3,err);
				connection.end();
			} else if (res.length == 0) {
				//用户名不存在
				//TODO role
				connection.query('insert into usr(name,role,password) value("' + name +'",0,"' + pwd + '")', function(err, res, fields) {
					if(err) {
						//数据库错误
						callback(-3,err);
						connection.end();
					} else {
						//注册成功
						callback(0);
						connection.end();
					}
				});
			} else {
				//用户名存在
				callback(-1);
				connection.end();
			}
		});
	},

	unregister: function(name, pwd, callback) {

	},
};