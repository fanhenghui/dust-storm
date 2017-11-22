const  mysql = require('mysql');
const fs = require('fs');
let db = null;

function connectDB() {
	fs.readFile(global.configPath, (err,data)=>{
		let configStr = data.toString();
		let configLines = configStr.split('\n');
		if (configLines.length < 5) {
			console.log('config file damadged.')
			return {};
		}
		let addr = {};
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

		db = mysql.createPool( {
			host:addr.ip,
			port:addr.port,
			user:addr.user,
			password:addr.password,
			database:addr.database,
		});
	});
}

connectDB();


module.exports = { 
	signIn: function(name, pwd, callback){
		if (db == null) {
			connectDB();
			callback(-3,"数据库连接失败");
			return;
		}

		db.query(`SELECT * FROM usr WHERE name='${name}'`, (err, res, fields)=>{
			if(err) {
				//数据库错误
				console.log(`DB query failed: ${err}`);
				callback(-3,"数据库错误");
			} else if(res.length == 0) {
				//用户名不存在
				callback(-1);
			} else {
				if (res[0].password == pwd ) {
					//登录成功
					callback(0);
				} else {
					//密码不正确
					callback(-2);
				} 
			}
		});
	},
	
	register: function(name, pwd, callback) {
		if (db == null) {
			connectDB();
			callback(-3,"数据库连接失败");
			return;
		}

		db.query(`SELECT * FROM usr WHERE name='${name}'`, (err, res, fields)=>{
			if(err) {
				//数据库错误
				callback(-3,err);
				connection.end();
			} else if (res.length == 0) {
				//用户名不存在
				//TODO role
				db.query(`INSERT INTO usr(name,role,password) VALUES ('${name}',0,'${pwd}')`, (err, res, fields)=>{
					if(err) {
						//数据库错误
						callback(-3,err);
					} else {
						//注册成功
						callback(0);
					}
				});
			} else {
				//用户名存在
				callback(-1);
			}
		});
	},

	unregister: function(name, pwd, callback) {

	},
};