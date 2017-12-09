const  mysql = require('mysql');
const fs = require('fs');
let pool = null;

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
			if (items.length == 3 && items[0] == 'DBIP')	 {
				addr['ip'] = items[2];
			} else if (items.length == 3 && items[0] == 'DBPort')	 {
				addr['port'] = items[2];
			} else if (items.length == 3 && items[0] == 'DBUser')	 {
				addr['user'] = items[2];
			} else if (items.length == 3 && items[0] == 'DBPWD')	 {
				addr['password'] = items[2];
			} else if (items.length == 3 && items[0] == 'DBName')	 {
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

		pool = mysql.createPool( {
			connectionLimit:10,//TODO，这里的值最好要和server带的用户数量匹配，估计N/2就够了
			host:addr.ip,
			port:addr.port,
			user:addr.user,
			password:addr.password,
			database:addr.database,
		});
		

		pool.on('release', function (connection) {
			console.log('Connection %d released', connection.threadId);
		});
	});
}

connectDB();


module.exports = { 
	closeDB: function name(params) {
		console.log('close DB');
		pool.end(err=>{
			if(err) {
				console.log(`DB error when close pool: ${err}`);
			}
		});
	},

	signIn: function(name, pwd, callback){
		if (pool == null) {
			connectDB();
			callback(-3,"数据库连接失败");
			return;
		}

		//TODO check name and pwd format
		pool.getConnection((err, connection) =>{
			if (err) {
				console.log(`DB connect error when sign in: ${err}`);
				callback(-4,"数据库错误");
			} else {
				connection.query(`SELECT * FROM usr WHERE name='${name}'`, (err, res, fields)=>{
					if(err) {
						//done with the connection
						connection.release();
						//database error
						console.log(`DB error when sign in: ${err}`);
						callback(-4,"数据库错误");
					} else if(res.length == 0) {
						//done with the connection
						connection.release();
						//user doesn't exist
						callback(-1);
					} else {
						console.log(res[0]);
						if (res[0].password != pwd ) {
							//done with the connection
							connection.release();
							//error password
							callback(-2);
						} else if (res[0].online[0] === 1) {
							//done with the connection
							connection.release();
							//repeat login
							callback(-3);
						} else {
							//update online 
							connection.query(`UPDATE usr SET online=1 WHERE name='${name}'`, (err, res, fields)=> {
								//done with the connection
								connection.release();
								if (err) {
									//database error
									console.log(`DB error when sign in: ${err}`);
									callback(-4,"数据库错误");
								} else {
									//login successs
									callback(0);
								}
							});
							
						}
					}
				});
			}
		});
		
	},
	
	register: function(name, pwd, callback) {
		if (pool == null) {
			connectDB();
			console.log(`DB error when register: ${err}`);
			callback(-4,"数据库连接失败");
			return;
		}
		pool.getConnection((err, connection) =>{
			if (err) {
				console.log(`DB connect error when sign in: ${err}`);
				callback(-4,"数据库错误");
			} else {
				connection.query(`SELECT * FROM usr WHERE name='${name}'`, (err, res, fields)=>{
					if(err) {
						//done with the connection
						connection.release();
						//database error
						callback(-4,err);
					} else if (res.length != 0) {
						//done with the connection
						connection.release();
						//username has exist
						callback(-1);
					} else {
						connection.query(`INSERT INTO usr(name,role,online,password) VALUES ('${name}',0,0,'${pwd}')`, (err, res, fields)=>{
							//done with the connection
							connection.release();
							if(err) {
								//database error
								console.log(`DB error when register: ${err}`);
								callback(-4,err);
							} else {
								//register success
								callback(0);
							}
						});
					}
				});
			}
		});
	},

	signOut: function(name) {
		if (pool == null) {
			connectDB();
			return;
		}
		pool.getConnection((err, connection) =>{
			if (err) {
				console.log(`DB connect error when sign in: ${err}`);
				callback(-4,"数据库错误");
			} else {
				connection.query(`SELECT * FROM usr WHERE name='${name}'`, (err, res, fields)=>{
					if(err) {
						//done with the connection
						connection.release();
						console.log(`DB error when sign out: ${err}`);
					} else {
						connection.query(`UPDATE usr SET online=0 WHERE name='${name}'`, (err, res, fields)=> {
							//done with the connection
							connection.release();
							if (err) {
								//database error
								console.log(`DB error when sign out: ${err}`);
							}
						});
					}
				});
			}
		});
	}
};