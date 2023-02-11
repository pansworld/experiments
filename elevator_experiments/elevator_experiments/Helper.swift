//
//  Helper.swift
//  elevator_experiments
//
//  Created by Pankaj Sharma on 1/27/23.
//

import Foundation
import MSAL

// Update the below to your client ID you received in the portal. The below is for running the demo only
let kClientID = "Your_Application_Id_Here"
let kGraphEndpoint = "https://graph.microsoft.com/" // the Microsoft Graph endpoint
let kAuthority = "https://login.microsoftonline.com/common" // this authority allows a personal Microsoft account and a work or school account in any organization's Azure AD tenant to sign in

let kScopes: [String] = ["user.read"] // request permission to read the profile of the signed-in user

var accessToken = String()
var applicationContext : MSALPublicClientApplication?
var webViewParameters : MSALWebviewParameters?
var currentAccount: MSALAccount?

struct MSALConfig {
    var url = "https://login.microsoftonline.com/common"
    var clientId = "3de9c462-dfd5-4082-a1af-b0e62822b8ab"
    var redirectURI = "msauth.pansworld.elevator-experiments://auth"
    var baseURL = "https://graph.microsoft.com/v1.0/me/drive/root:/"
    var authorityURL = "https://login.microsoftonline.com/common"
}



class Helpers {
    var retVal: Any
    var accessToken: String
    var accountId: String
    var accountName: String
    let msalConfig = MSALConfig()
    let getDaysSince1900 = getDaysSince190
    var excelId: String
    var tableRow: [[Any]]
    var el1Prob: String
    var el2Prob: String
    var el3Prob: String
    var lastIndex: Int
    
    init(){
        //Constructor
        self.retVal = ""
        self.accessToken = ""
        self.accountId = ""
        self.accountName = ""
        self.excelId = ""
        self.tableRow = [[]]
        self.el1Prob = ""
        self.el2Prob = ""
        self.el3Prob = ""
        self.lastIndex = 0

    }
    
    func addData(input: String) {
        let hour = Calendar.current.component(.hour, from: Date())
        let day = Calendar.current.component(.day, from: Date())
        let month = Calendar.current.component(.month, from: Date())
        let year = Calendar.current.component(.year, from: Date())

        var logFile: URL? {
            guard let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return nil }
            //print(documentsDirectory)
            let fileName = "data.csv"
            return documentsDirectory.appendingPathComponent(fileName)
        }
        
        guard let logFile = logFile else {
            return
        }

        guard let data = ("\(month)/\(day)/\(year),\(hour),\(input)\n").data(using: String.Encoding.utf8) else { return }

        if FileManager.default.fileExists(atPath: logFile.path) {
            if let fileHandle = try? FileHandle(forWritingTo: logFile) {
                fileHandle.seekToEndOfFile()
                fileHandle.write(data)
                fileHandle.closeFile()
            }
        } else {
            var csvText = "Date,Hour,Elevator\n"


                 let newLine = "\(month)/\(day)/\(year),\(hour),\(input)\n"
                 csvText.append(newLine)


             do {
                try csvText.write(to: logFile, atomically: true, encoding: String.Encoding.utf8)

             } catch {
                 print("Failed to create file")
                 print("\(error)")
             }
             print(logFile)
        }
    }
    
    /**
         Get the number of dats since 1900
         Return the number of days as integer
     **/
    func getDaysSince190(inDate: Date) -> Int {
        let startDateString = "01/01/1900"
        
        // Create Date Formatter
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "M/d/yyyy"
        
        let days = Calendar.current.dateComponents([.day], from: dateFormatter.date(from: startDateString)!, to: inDate).day!
        
        return days
    }

    
    /**
        Check if we have an access token
     **/
    func isLoggedIn() -> Bool {
        if (self.accessToken == ""){
            return false
        }else{
            return true
        }
    }
    
    /**
        Get the access token
        Asynchronous function
     **/
    func getAccessToken(controller: UIViewController) {

        do {
            let authority = try MSALAuthority(url: URL(string: msalConfig.url)!)
            let pcaConfig = MSALPublicClientApplicationConfig(clientId: msalConfig.clientId, redirectUri: msalConfig.redirectURI, authority: authority)
            let application = try MSALPublicClientApplication(configuration: pcaConfig)
            let webViewParameters = MSALWebviewParameters(authPresentationViewController: controller)
            let interactiveParameters = MSALInteractiveTokenParameters(scopes: ["user.read","Files.ReadWrite.All"], webviewParameters: webViewParameters)
            interactiveParameters.promptType = .selectAccount
                application.acquireToken(with: interactiveParameters){ (result, error) in
                //print("RESULT = \(String(describing: result))")
 
                guard let result = result else {
                    print("ERROR=\(String(describing: error))")
                    return
                }
                
                self.accessToken = result.accessToken
 
            }
        }catch{
            return
        }
    }
    
    /**
        Get the last row in the excel table
        Asynchronous function
    **/
    func getLastExcelTableRow(excelFile: String, tableName: String) async {
        do {
            var graphURI = msalConfig.baseURL + excelFile + ":/workbook/worksheets" //"https://graph.microsoft.com/v1.0/me/drive/root:/elevator_experiments.xlsx:/workbook/worksheets"
            var url = URL(string: graphURI)
            var request = URLRequest(url: url!)

            // Set the Authorization header for the request. We use Bearer tokens, so we specify Bearer + the token we got from the result
            request.setValue("Bearer \(self.accessToken)", forHTTPHeaderField: "Authorization")


            //Check if we have the excel id if not get it
            if (self.excelId == ""){
                let (data, _) = try await URLSession.shared.data(for: request)
                let result = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
                
                let json = result!["value"] as? [[String: Any]]
                self.excelId = String(describing: json?[0]["id"])
                self.excelId = String(self.excelId.dropLast(2))
                self.excelId = String(self.excelId.dropFirst(10))
                
            }
            

            if (self.excelId != ""){
                graphURI = msalConfig.baseURL + excelFile + ":/workbook/worksheets(%27%7B" + String(describing: self.excelId) + "%7D%27)/tables/" + tableName + "/rows"
                 url = URL(string: graphURI)
                 request = URLRequest(url: url!)
                // Set the Authorization header for the request. We use Bearer tokens, so we specify Bearer + the token we got from the result
                request.setValue("Bearer \(self.accessToken)", forHTTPHeaderField: "Authorization")
                let (data, _) = try await URLSession.shared.data(for: request)
                let result = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
                
                let json = result!["value"] as? [[String: Any]]
                self.retVal = String(describing: json?[0]["id"])
                self.tableRow = ((json?[json!.count - 1]["values"])  as! [[Any]])
                self.lastIndex = json!.count - 1
                
            }

        }catch{
            //TODO: What happens if there is an error
        }
        
    }
    
    /**
        Process the row
        Update or add based on the days method
     **/
    func processTableRow(excelFile: String, tableName: String, elevatorNo: Int) async {
        
        //print(self.accessToken)
        //Get the last row
        await self.getLastExcelTableRow(excelFile: excelFile, tableName: tableName)
        
        let lastRow = self.tableRow
        let curDate = Date()
        
        //Create a date formatter
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "M/d/yyyy"
        
        
        let days = self.getDaysSince190(inDate: curDate)
        //print(days)
        //print(lastRow[0][0])
        
        //Check the current date and days
        //If the current days do not match then add record
        if (lastRow[0][0] as! Int != days){
            //print("Insert")
            await self.appendExcelTableRow(excelFile: excelFile, tableName: tableName, elevatorNo: elevatorNo, days: days)
        }else{
            //If the current days match then update record
            //print("Update")
            await self.updateExcelTableRow(excelFile: excelFile, tableName: tableName, data: lastRow[0], elevatorNo: elevatorNo)
       }

    }
    
    /**
        Update the row and index in the excel table
        Asynchronous function
     **/
    func updateExcelTableRow(excelFile: String, tableName: String, data: [Any], elevatorNo: Int) async {
        do {
            let graphURI = msalConfig.baseURL + excelFile + ":/workbook/worksheets(%27%7B" + String(describing: self.excelId) + "%7D%27)/tables/" + tableName + "/rows/$/ItemAt%28index=\(self.lastIndex)%29"
            let url = URL(string: graphURI)
            var request = URLRequest(url: url!)
            
            // Set the Authorization header for the request. We use Bearer tokens, so we specify Bearer + the token we got from the result
            request.setValue("Bearer \(self.accessToken)", forHTTPHeaderField: "Authorization")
           
            var dataBody = [data[0],data[1],data[2],data[3],nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil]
            
            dataBody[elevatorNo]=dataBody[elevatorNo] as! Int + 1
            
            let postData = ["values": [dataBody]]
            request.httpMethod = "PATCH"
            request.setValue("application/json; charset=utf-8", forHTTPHeaderField: "Content-Type")  // the request is JSON
            
            let jsonData = try? JSONSerialization.data(withJSONObject: postData)
            request.httpBody = jsonData
            
            let (data, _) = try await URLSession.shared.data(for: request)
            //print(data)
            let result = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
            
            //print(result!["values"])
            let json = result!["values"] as? [[Any]]
            helper.el1Prob=String(format: "%.2f", json?[0][13] as! Double)
            helper.el2Prob=String(format: "%.2f", json?[0][14] as! Double)
            helper.el3Prob=String(format: "%.2f", json?[0][15] as! Double)
            
        }catch{
            
        }

        
        return
    }
    
    /**
        Append the row to the excel table
        Asynchronous function
     **/
    func appendExcelTableRow(excelFile: String, tableName: String, elevatorNo: Int, days: Int) async  {
        
        do {
            let graphURI = msalConfig.baseURL + excelFile + ":/workbook/worksheets(%27%7B" + String(describing: self.excelId) + "%7D%27)/tables/" + tableName + "/rows"
            let url = URL(string: graphURI)
            var request = URLRequest(url: url!)
            
            // Set the Authorization header for the request. We use Bearer tokens, so we specify Bearer + the token we got from the result
            request.setValue("Bearer \(self.accessToken)", forHTTPHeaderField: "Authorization")
           
            var dataBody = [days,0,0,0,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil]
            
            dataBody[elevatorNo]=1
            
            let postData = ["values": [dataBody]]
            request.httpMethod = "POST"
            request.setValue("application/json; charset=utf-8", forHTTPHeaderField: "Content-Type")  // the request is JSON
            
            let jsonData = try? JSONSerialization.data(withJSONObject: postData)
            request.httpBody = jsonData
            
            let (data, _) = try await URLSession.shared.data(for: request)
            print(data)
            let result = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any]
            
            //print(result!["values"])
            let json = result!["values"] as? [[Any]]
            helper.el1Prob=String(format: "%.2f", json?[0][13] as! Double)
            helper.el2Prob=String(format: "%.2f", json?[0][14] as! Double)
            helper.el3Prob=String(format: "%.2f", json?[0][15] as! Double)

        }catch{
            
        }
        
        
        return
    }
    
    
    /**func executeGetRequest(graphURI :String, accessToken :String) async throws {
        let url = URL(string: graphURI)
        var request = URLRequest(url: url!)

        // Set the Authorization header for the request. We use Bearer tokens, so we specify Bearer + the token we got from the result
        request.setValue("Bearer \(accessToken)", forHTTPHeaderField: "Authorization")
        
        let task = URLSession.shared.dataTask(with: request) { data, response, error in
            
            if let error = error {
                print("Couldn't get graph result: \(error)")
                return
            }
            
            if let result1 = try? JSONSerialization.jsonObject(with: data!, options: []) as? [String: Any]
            {
                
                if result1["value"] is [String: Any]{
                    //print("result = \(id)")
                }else{
                    //print("Original = \(result1)")
                    let json = result1.first?.value as? [[String: Any]]
                    
                    let id = json?[0]["id"]
                    if id == nil{
                        print("There is no value")
                        self.retVal = "No Id Found"
                    }else{
                        print("There is a value")
                        self.retVal = id as! String
                    }
                    return

                }
            }  else {
                print("Couldn't deserialize result JSON")
                return
            }
            
        }
        task.resume()
    }


    func getWorkBookId(accessToken :String) async -> String {

        //Get the list of drives
        //
        let graphURI = "https://graph.microsoft.com/v1.0/me/drive/root/search(q='elevator_experiments.xlsx')?select=name,id,webUrl"
        
        do{
            try await self.executeGetRequest(graphURI: graphURI , accessToken: accessToken)
            print ("Return Value \(self.retVal)")
            return self.retVal as! String
        }catch {
            return self.retVal as! String
        }

    }**/

    
}

