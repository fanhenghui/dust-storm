var mySQL = require('mysql');
var dbAdd = require("./db-address");

module.exports = { 
	signIn: function(name, pwd, callback){
		var connection = mySQL.createConnection( {
			host:dbAdd.address.ip,
			port:dbAdd.address.port,
			user:dbAdd.address.user,
			password: dbAdd.address.password,
			database:dbAdd.address.database,
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
		var connection = mySQL.createConnection( {
			host:dbAdd.address.ip,
			port:dbAdd.address.port,
			user:dbAdd.address.user,
			password: dbAdd.address.password,
			database:dbAdd.address.database,
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