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

		//TODO check name and pwd

		db.query(`SELECT * FROM usr WHERE name='${name}'`, (err, res, fields)=>{
			if(err) {
				//database error
				console.log(`DB error when sign in: ${err}`);
				callback(-4,"数据库错误");
			} else if(res.length == 0) {
				//user doesn't exist
				callback(-1);
			} else {
				console.log(res[0]);
				if (res[0].password != pwd ) {
					//error password
					callback(-2);
				} else if (res[0].online[0] === 1) {
					//repeat login
					callback(-3);
				} else {
					//update online 
					db.query(`UPDATE usr SET online=1 WHERE name='${name}'`, (err, res, fields)=> {
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
	},
	
	register: function(name, pwd, callback) {
		if (db == null) {
			connectDB();
			console.log(`DB error when register: ${err}`);
			callback(-4,"数据库连接失败");
			return;
		}

		db.query(`SELECT * FROM usr WHERE name='${name}'`, (err, res, fields)=>{
			if(err) {
				//database error
				callback(-4,err);
			} else if (res.length == 0) {
				db.query(`INSERT INTO usr(name,role,online,password) VALUES ('${name}',0,0,'${pwd}')`, (err, res, fields)=>{
					if(err) {
						//database error
						console.log(`DB error when register: ${err}`);
						callback(-4,err);
					} else {
						//register success
						callback(0);
					}
				});
			} else {
				//username has exist
				callback(-1);
			}
		});
	},

	signOut: function(name) {
		if (db == null) {
			connectDB();
			return;
		}
		db.query(`SELECT * FROM usr WHERE name='${name}'`, (err, res, fields)=>{
			if(err) {
				console.log(`DB error when sign out: ${err}`);
			} else {
				db.query(`UPDATE usr SET online=0 WHERE name='${name}'`, (err, res, fields)=> {
					if (err) {
						//database error
						console.log(`DB error when sign out: ${err}`);
					}
				});
			}
		});
	}
};